import cv2
import copy
import json
import math
import os
import sys
import time
from typing import Iterable
import numpy as np
from shapely.geometry import Polygon
import torch
from util.misc import nested_tensor_from_tensor_list
import util.misc as utils
from contextlib import redirect_stdout
from s3d_floorplan_eval.Evaluator.Evaluator import Evaluator
from s3d_floorplan_eval.options import MCSSOptions
from s3d_floorplan_eval.DataRW.S3DRW import S3DRW
from s3d_floorplan_eval.DataRW.wrong_annotatios import wrong_s3d_annotations_list
from scenecad_eval.Evaluator import Evaluator_SceneCAD
from util.poly_ops import pad_gt_polys,pad_gt_polys_to_edges,get_gt_polys
from util.plot_utils import plot_room_map, plot_score_map, plot_floorplan_with_regions, plot_semantic_rich_floorplan,plot_room_map_with_edges,plot_floorplan_with_edges
from util.edge_utils import remove_short_edges,get_corners_from_edges,remove_duplicate_corners,merge_points,refine_rooms,remove_multi_polygon,remove_rooms_with_iou
options = MCSSOptions()
opts = options.parse()

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, max_norm: float = 0, args = None):
    model.train()
    criterion.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    metric_logger.add_meter('grad_norm', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20
    
    for batched_inputs in metric_logger.log_every(data_loader, print_freq, header):
        samples = [x["image"].to(device) for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        room_targets = pad_gt_polys_to_edges(gt_instances, model.num_queries_per_poly, device)

        targets = get_gt_polys(gt_instances, model.num_queries_per_poly, device)

        dn_args = (targets, args.scalar, args.label_noise_scale, args.poly_noise_scale)

        outputs,mask_dict = model(samples,dn_args)
        loss_dict = criterion(outputs, room_targets,mask_dict)
        weight_dict = criterion.weight_dict
        losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        losses_reduced_scaled = sum(loss_dict_scaled.values())

        loss_value = losses_reduced_scaled.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if max_norm > 0:
            grad_total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        else:
            grad_total_norm = utils.get_total_grad_norm(model.parameters(), max_norm)
        optimizer.step()

        metric_logger.update(loss=loss_value, **loss_dict_scaled, **loss_dict_unscaled)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        metric_logger.update(grad_norm=grad_total_norm)

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(model, criterion, dataset_name, data_loader, device,epoch = None):

    model.eval()
    criterion.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    for batched_inputs in metric_logger.log_every(data_loader, 10, header):

        samples = [x["image"].to(device) for x in batched_inputs]
        
        scene_ids = [x["image_id"]for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]
        room_targets = pad_gt_polys_to_edges(gt_instances, model.num_queries_per_poly, device)


        outputs,_ = model(samples)
        loss_dict = criterion(outputs, room_targets,None,scene_ids,epoch)
        weight_dict = criterion.weight_dict
        weight_dict['loss_coords']=5
        weight_dict['tgt_loss_coords'] = 5

        bs = outputs['pred_logits'].shape[0]
        pred_logits = outputs['pred_logits']
        pred_corners = outputs['pred_coords']
        pred_logits = torch.sigmoid(pred_logits)

        fg_mask = pred_logits> 0.5 # select valid corners

        if 'pred_room_logits' in outputs:
            prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
            _, pred_room_label = prob[..., :-1].max(-1)


        # process per scene
        for i in range(bs):

            if dataset_name == 'stru3d':
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode = "online_eval")
                evaluator = Evaluator(curr_data_rw, curr_opts)
            elif dataset_name == 'scenecad':
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1,2).astype(np.int)]
                evaluator = Evaluator_SceneCAD()
            
            print("Running Evaluation for scene %s" % scene_ids[i])

            fg_mask_per_scene = fg_mask[i]
            pred_corners_per_scene = pred_corners[i]
            pred_logits_per_scene = pred_logits[i]

            room_polys = []
            room_edges = []
            
            semantic_rich = 'pred_room_logits' in outputs
            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []
                pred_room_label_per_scene = pred_room_label[i].cpu().numpy()

            # process per room
            for j in range(fg_mask_per_scene.shape[0]):
                fg_mask_per_room = fg_mask_per_scene[j]
                pred_corners_per_room = pred_corners_per_scene[j]
                pred_logits_per_room = pred_logits_per_scene[j][fg_mask_per_room].cpu().numpy()

                valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]
                if len(valid_corners_per_room)>0:
                    corners = (valid_corners_per_room * 255).cpu().numpy()
                    edges = corners
                    corners,filtered_pred_logits = remove_short_edges(corners,pred_logits_per_room)
                    corners = get_corners_from_edges(corners,filtered_pred_logits)

                    corners = remove_duplicate_corners(corners)
                    corners = np.around(corners).astype(np.int32)
                    corners = merge_points(corners,2)
                    edges = np.around(edges).astype(np.int32)

                    if not semantic_rich:
                        # only regular rooms
                        if len(corners)>=4 and Polygon(corners).area >= 100:
                                room_polys.append(corners)
                                room_edges.append(edges)
                    else:
                        # regular rooms
                        if pred_room_label_per_scene[j] not in [16,17]:
                            if len(corners)>=4 and Polygon(corners).area >= 100:
                                room_polys.append(corners)
                                room_types.append(pred_room_label_per_scene[j])
                        # window / door
                        elif len(corners)==2:
                            window_doors.append(corners)
                            window_doors_types.append(pred_room_label_per_scene[j])
            overlap=False
            shapely_polygons = []
            for np_array in room_polys:
                points = [tuple(point) for point in np_array]                
                if len(points) > 0 and points[0] != points[-1]:
                    points.append(points[0])
                shapely_poly = Polygon(points)
                shapely_polygons.append(shapely_poly)
            try:
                shapely_polygons = remove_multi_polygon(shapely_polygons)
                shapely_polygons = remove_rooms_with_iou(shapely_polygons)
                polygon_list,overlap = refine_rooms(shapely_polygons,overlap)
                room_ = []
                for polygon in polygon_list:
                    room = np.array(polygon.exterior.coords, dtype=np.int32)[:-1]
                    room_.append(room)

                room_polys=room_
            except:
                room_polys=room_polys
   
            
            if dataset_name == 'stru3d':
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                                                            room_polys=room_polys, 
                                                            room_types=room_types, 
                                                            window_door_lines=window_doors, 
                                                            window_door_lines_types=window_doors_types)
            elif dataset_name == 'scenecad':
                quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)

            if 'room_iou' in quant_result_dict_scene:
                metric_logger.update(room_iou=quant_result_dict_scene['room_iou'])
            
            metric_logger.update(room_prec=quant_result_dict_scene['room_prec'])
            metric_logger.update(room_rec=quant_result_dict_scene['room_rec'])
            metric_logger.update(corner_prec=quant_result_dict_scene['corner_prec'])
            metric_logger.update(corner_rec=quant_result_dict_scene['corner_rec'])
            metric_logger.update(angles_prec=quant_result_dict_scene['angles_prec'])
            metric_logger.update(angles_rec=quant_result_dict_scene['angles_rec'])

            if semantic_rich:
                metric_logger.update(room_sem_prec=quant_result_dict_scene['room_sem_prec'])
                metric_logger.update(room_sem_rec=quant_result_dict_scene['room_sem_rec'])
                metric_logger.update(window_door_prec=quant_result_dict_scene['window_door_prec'])
                metric_logger.update(window_door_rec=quant_result_dict_scene['window_door_rec'])

        loss_dict_scaled = {k: v * weight_dict[k]
                                    for k, v in loss_dict.items() if k in weight_dict}
        loss_dict_unscaled = {f'{k}_unscaled': v
                                      for k, v in loss_dict.items()}
        metric_logger.update(loss=sum(loss_dict_scaled.values()),
                             **loss_dict_scaled,
                             **loss_dict_unscaled)

    print("Averaged stats:", metric_logger)

    stats = {k: meter.global_avg for k, meter in metric_logger.meters.items()}

    return stats

@torch.no_grad()
def evaluate_floor(model, dataset_name, data_loader, device, output_dir, plot_pred=True, plot_density=True, plot_gt=True, semantic_rich=False):
    model.eval()
    time_all = []
    quant_result_dict = None
    scene_counter = 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)  
    for batched_inputs in data_loader:

        samples = [x["image"].to(device) for x in batched_inputs]
        scene_ids = [x["image_id"] for x in batched_inputs]
        gt_instances = [x["instances"].to(device) for x in batched_inputs]

      
        # draw GT map
        if plot_gt:
            for i, gt_inst in enumerate(gt_instances):
                if not semantic_rich:
                    # plot regular room floorplan
                    gt_polys = []
                    density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
                    density_map = np.repeat(density_map, 3, axis=2)
                    for j, poly in enumerate(gt_inst.gt_masks.polygons):
                        corners = poly[0].reshape(-1, 2)
                        gt_polys.append(corners)
                        
                    gt_room_polys = [np.array(r) for r in gt_polys]
                    gt_floorplan_map = plot_floorplan_with_regions(gt_room_polys, scale=1000)
                    cv2.imwrite(os.path.join(output_dir, '{}_gt.png'.format(scene_ids[i])), gt_floorplan_map)
                else:
                    # plot semantically-rich floorplan
                    gt_sem_rich = []
                    for j, poly in enumerate(gt_inst.gt_masks.polygons):
                        corners = poly[0].reshape(-1, 2).astype(np.int)
                        corners_flip_y = corners.copy()
                        corners_flip_y[:,1] = 255 - corners_flip_y[:,1]
                        corners = corners_flip_y
                        gt_sem_rich.append([corners, gt_inst.gt_classes.cpu().numpy()[j]])

                    gt_sem_rich_path = os.path.join(output_dir, '{}_sem_rich_gt.png'.format(scene_ids[i]))
                    plot_semantic_rich_floorplan(gt_sem_rich, gt_sem_rich_path, prec=1, rec=1) 


        start_time = time.time()
        outputs,_ = model(samples)
        end_time = time.time()
        inference_time = (end_time - start_time) * 1000 

        time_all.append(inference_time)
        pred_logits = outputs['pred_logits']
        pred_corners = outputs['pred_coords']
        pred_logits = torch.sigmoid(pred_logits)
        fg_mask = pred_logits > 0.5 # select valid corners

        if 'pred_room_logits' in outputs:
            prob = torch.nn.functional.softmax(outputs['pred_room_logits'], -1)
            _, pred_room_label = prob[..., :-1].max(-1)

        # process per scene
        for i in range(pred_logits.shape[0]):
            
            if dataset_name == 'stru3d':
                if int(scene_ids[i]) in wrong_s3d_annotations_list:
                    continue
                curr_opts = copy.deepcopy(opts)
                curr_opts.scene_id = "scene_0" + str(scene_ids[i])
                curr_data_rw = S3DRW(curr_opts, mode = "test")
                evaluator = Evaluator(curr_data_rw, curr_opts)
            elif dataset_name == 'scenecad':
                gt_polys = [gt_instances[i].gt_masks.polygons[0][0].reshape(-1,2).astype(np.int)]
                evaluator = Evaluator_SceneCAD()

            print("Running Evaluation for scene %s" % scene_ids[i])

            fg_mask_per_scene = fg_mask[i]
            pred_logits_per_scene = pred_logits[i]
            pred_corners_per_scene = pred_corners[i]
            room_polys = []
            room_edges = []
            room_edge_lengths=[]

            if semantic_rich:
                room_types = []
                window_doors = []
                window_doors_types = []
                pred_room_label_per_scene = pred_room_label[i].cpu().numpy()

            # process per room
            for j in range(fg_mask_per_scene.shape[0]):

                    fg_mask_per_room = fg_mask_per_scene[j]

                    pred_logits_per_room = pred_logits_per_scene[j][fg_mask_per_room].cpu().numpy()
                    
                    pred_corners_per_room = pred_corners_per_scene[j]

                    valid_corners_per_room = pred_corners_per_room[fg_mask_per_room]

                    if len(valid_corners_per_room)>0:
                        corners = (valid_corners_per_room * 255).cpu().numpy()
                        edges = corners
                        corners,pred_logits_per_room = remove_short_edges(corners,pred_logits_per_room)
                        corners = np.around(corners).astype(np.int32)
                        corners = get_corners_from_edges(corners,pred_logits_per_room,10)
                        corners = np.around(corners).astype(np.int32)
                        corners = merge_points(corners,2)
                        edges = np.around(edges).astype(np.int32)


                        if not semantic_rich:
                            # only regular rooms
                            if len(corners)>=4 :
                                    room_polys.append(corners)
                                    room_edges.append(edges)


                        else:
                            # regular rooms
                            if pred_room_label_per_scene[j] not in [16,17]:
                                if len(corners)>=4 and Polygon(corners).area >= 100:
                                    room_polys.append(corners)

                                    room_types.append(pred_room_label_per_scene[j])
                            # window / door
                            elif len(corners)==2:
                                window_doors.append(corners)
                                window_doors_types.append(pred_room_label_per_scene[j])

            overlap=False
            shapely_polygons = []
            for np_array in room_polys:

                points = [tuple(point) for point in np_array]

                if len(points) > 0 and points[0] != points[-1]:
                    points.append(points[0])
                
                shapely_poly = Polygon(points)
                shapely_polygons.append(shapely_poly)
            try:

                shapely_polygons = remove_rooms_with_iou(shapely_polygons)
                polygon_list,overlap = refine_rooms(shapely_polygons,overlap)

                room_ = []
                for polygon in polygon_list:
       
                    room = np.array(polygon.exterior.coords, dtype=np.int32)[:-1]
                    room_.append(room)


                room_polys=room_
            except:
                room_polys=room_polys


            if dataset_name == 'stru3d':
                if not semantic_rich:
                    quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys)
                else:
                    quant_result_dict_scene = evaluator.evaluate_scene(
                                                            room_polys=room_polys, 
                                                            room_types=room_types, 
                                                            window_door_lines=window_doors, 
                                                            window_door_lines_types=window_doors_types)
    
            elif dataset_name == 'scenecad':
                quant_result_dict_scene = evaluator.evaluate_scene(room_polys=room_polys, gt_polys=gt_polys)

            if quant_result_dict is None:
                quant_result_dict = quant_result_dict_scene
            else:
                for k in quant_result_dict.keys():
                    quant_result_dict[k] += quant_result_dict_scene[k]

            scene_counter += 1

            if plot_pred:
                if semantic_rich:
                    # plot predicted semantic rich floorplan
                    pred_sem_rich = []
                    for j in range(len(room_polys)):
                        temp_poly = room_polys[j]
                        temp_poly_flip_y = temp_poly.copy()
                        temp_poly_flip_y[:,1] = 255 - temp_poly_flip_y[:,1]
                        pred_sem_rich.append([temp_poly_flip_y, room_types[j]])
                    for j in range(len(window_doors)):
                        temp_line = window_doors[j]
                        temp_line_flip_y = temp_line.copy()
                        temp_line_flip_y[:,1] = 255 - temp_line_flip_y[:,1]
                        pred_sem_rich.append([temp_line_flip_y, window_doors_types[j]])

                    pred_sem_rich_path = os.path.join(output_dir, '{}_sem_rich_pred.png'.format(scene_ids[i]))
                    plot_semantic_rich_floorplan(pred_sem_rich, pred_sem_rich_path, prec=quant_result_dict_scene['room_prec'], rec=quant_result_dict_scene['room_rec'])
                else:
                    # plot regular room floorplan
                    room_polys = [np.array(r) for r in room_polys]
                    floorplan_map = plot_floorplan_with_regions(room_polys, scale=1000)
                    cv2.imwrite(os.path.join(output_dir, '{}_pred_floorplan.png'.format(scene_ids[i])), floorplan_map)

            room_edges = [np.array(r) for r in room_edges]
            density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
            edge_map = plot_floorplan_with_edges(room_edges, scale=1000,density_map = density_map)

            cv2.imwrite(os.path.join(output_dir, '{}_pred_edge.png'.format(scene_ids[i])), edge_map)
            density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
            density_map = np.repeat(density_map, 3, axis=2)
            pred_room_map = np.zeros([256, 256, 3])

            for room_poly in room_edges:
                pred_room_map = plot_room_map_with_edges(room_poly, pred_room_map)


            if plot_density:
                density_map = np.transpose((samples[i] * 255).cpu().numpy(), [1, 2, 0])
                density_map = np.repeat(density_map, 3, axis=2)
                pred_room_map = np.zeros([256, 256, 3])

                for room_poly in room_polys:
                    pred_room_map = plot_room_map(room_poly, pred_room_map)

                # plot predicted polygon overlaid on the density map
                pred_room_map = np.clip(pred_room_map + density_map, 0, 255)
                cv2.imwrite(os.path.join(output_dir, '{}_pred_room_map.png'.format(scene_ids[i])), pred_room_map)

    for k in quant_result_dict.keys():
        quant_result_dict[k] /= float(scene_counter)

    metric_category = ['room','corner','angles']
    if semantic_rich:
        metric_category += ['room_sem','window_door']
    for metric in metric_category:
        prec = quant_result_dict[metric+'_prec']
        rec = quant_result_dict[metric+'_rec']
        f1 = 2*prec*rec/(prec+rec)
        quant_result_dict[metric+'_f1'] = f1

    print("*************************************************")
    print(quant_result_dict)
    print("*************************************************")
    avg_time = sum(time_all) / len(time_all)
    print(f"Average inference time: {avg_time:.2f} ms")

    with open(os.path.join(output_dir, 'results.txt'), 'w') as file:
        file.write(json.dumps(quant_result_dict))
