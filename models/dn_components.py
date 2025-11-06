# ------------------------------------------------------------------------------------
# Modified from DN-DETR (https://github.com/IDEA-Research/DN-DETR)
# ------------------------------------------------------------------------------------
import cv2

import torch
import numpy as np

import matplotlib.pyplot as plt
from datasets import build_dataset

from models.losses import custom_L1_loss, dn_L1_loss,dn_angle_loss
from util.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)

import torch.nn.functional as F
from torch import nn
from util.poly_ops import pad_gt_polys, get_gt_polys




def prepare_for_dn(dn_args, tgt_weight, embedweight, batch_size, training, num_queries, num_classes, hidden_dim, label_enc):
    """
    :param dn_args: targets, scalar(number of dn groups), label_noise_scale, poly_noise_scale, 
    :param tgt_weight: learnbal tgt 
    :param embedweight: positional anchor queries
    :param batch_size: bs
    :param training: if it is training or inference
    :param num_queries: number of queires
    :param num_classes: number of classes
    :param hidden_dim: transformer hidden dim
    :param label_enc: encode labels in dn 
    :return:
    """

    if training:
        targets, scalar, label_noise_scale, poly_noise_scale = dn_args

    
    if tgt_weight is not None and embedweight is not None:
 
        indicator0 = torch.zeros([num_queries, 1]).cuda()
        # sometimes the target is empty, add a zero part of label_enc to avoid unused parameters
        tgt = torch.cat([tgt_weight, indicator0], dim=1) + label_enc.weight[0][0] * torch.tensor(0).cuda()
        # latent query
        refpoint_emb = embedweight  # [800,4]
    else:
        tgt = None
        refpoint_emb = None

    
    if training:
        
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets] 
        # The number of edges of a scene
        known_num = [sum(k) for k in known]
        points_num_each_poly = [t['lengths'] for t in targets] 
        # edge labels
        labels = torch.cat([t['labels'] for t in targets]) #[edge_num_allbatch]
        coords = torch.cat([t['coords'].reshape(-1,4) for t in targets])#[poly_num_allbatch,4]

        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)])

        known_lengths = torch.cat([t['lengths'] for t in targets])

        # add noise
        know_idx = [torch.nonzero(t) for t in known]  # [[poly_num*40,2]]

        unmask_poly = unmask_label = torch.cat(known)  # [poly_num_allbatch,40]
        known_indice = torch.nonzero(unmask_label + unmask_poly)
        known_indice = known_indice.view(-1)
        num_vis_poly = coords.shape[0]
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_coords = coords.repeat(scalar, 1)
        known_labels_expaned = known_labels.clone()
        known_coords_expand = known_coords.clone()

        known_lengths = known_lengths.repeat(scalar,1).view(-1)

        # noise on the label
        if label_noise_scale > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_scale)).view(-1)  # usually half of poly noise
            new_label = torch.randint_like(chosen_indice, 0, num_classes)  # randomly put a new one here
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
        # noise on the polygon
        if poly_noise_scale > 0:
            diff = torch.zeros_like(known_coords_expand)
            diff = known_coords_expand

            known_coords_expand += torch.mul((torch.rand_like(known_coords_expand) * 2 - 1.0),
                                           diff).cuda() * poly_noise_scale

            known_coords_expand = known_coords_expand.clamp(min=0.0, max=1.0)

        m = known_labels_expaned.long().to('cuda')

        input_label_embed = label_enc(m) 
        # add dn part indicator
        indicator1 = torch.ones([input_label_embed.shape[0], 1]).cuda() 
        input_label_embed = torch.cat([input_label_embed, indicator1], dim=1) 

        input_coords_embed = inverse_sigmoid(known_coords_expand)
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_coords = torch.zeros(pad_size, 4).cuda() 
        if tgt is not None and refpoint_emb is not None:
            input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
            input_query_coords = torch.cat([padding_coords, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
        else:
            input_query_label = padding_label.repeat(batch_size, 1, 1)
            input_query_coords = padding_coords.repeat(batch_size, 1, 1)

        # map in order
        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])

            map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_coords[(known_bid.long(), map_known_indice)] = input_coords_embed.float()

        tgt_size = pad_size + num_queries  
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0

        # make sure latent query cannot see the purturbed query
        attn_mask[pad_size:, :pad_size] = True
        # make sure purturbed cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_polys': (known_labels, known_coords),
            'know_idx': know_idx,
            'pad_size': pad_size,
            "known_lengths":known_lengths
        }
    else:  # no dn for inference
        if tgt is not None and refpoint_emb is not None:
            input_query_label = tgt.repeat(batch_size, 1, 1)
            input_query_coords = refpoint_emb.repeat(batch_size, 1, 1)
        else:
            input_query_label = None
            input_query_coords = None
        attn_mask = None
        mask_dict = None

    return input_query_label, input_query_coords, attn_mask, mask_dict


def dn_post_process(outputs_class, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_polys']=(output_known_class,output_known_coord)
    return outputs_class, outputs_coord


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    Returns:

    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_polys']
    known_labels, known_polys = mask_dict['known_lbs_polys']
    map_known_indice = mask_dict['map_known_indice']

    known_indice = mask_dict['known_indice']
    known_lengths = mask_dict['known_lengths']

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice]

    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
    num_tgt = known_indice.numel() 

    return known_labels, known_polys, output_known_class, output_known_coord, known_lengths


def tgt_loss_polys(src_polys, tgt_polys,target_len,angle):
    """Compute the losses related to the polygons"""
    if len(tgt_polys) == 0:
        return {
            'tgt_loss_coords': torch.as_tensor(0.).to('cuda')
        }
    loss_coords,loss_angles = dn_L1_loss(src_polys, tgt_polys, target_len,angle=angle)

    losses = {}
    losses['tgt_loss_coords'] = loss_coords
    if angle:
        losses['tgt_loss_angles'] = loss_angles
    return losses


def tgt_loss_labels(src_logits_, tgt_labels_, log=True):
    """Classification loss (NLL)"""
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
        }

    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0).float()

    loss_ce = F.binary_cross_entropy_with_logits(src_logits, tgt_labels)

    losses = {'tgt_loss_ce': loss_ce}

    return losses


def compute_dn_loss(mask_dict, training, aux_num,angle):
    """
       compute dn loss in criterion
       Args:
           mask_dict: a dict for dn information
           training: training or inference flag
           aux_num: aux loss number
           focal_alpha:  for focal loss
       """
    losses = {}
    if training and 'output_known_lbs_polys' in mask_dict:
        known_labels, known_polys, output_known_class, output_known_coord, known_lengths = prepare_for_loss(mask_dict)
        losses.update(tgt_loss_labels(output_known_class[-1].view(-1), known_labels))
        losses.update(tgt_loss_polys(output_known_coord[-1], known_polys, known_lengths,angle))
    else:
        losses['tgt_loss_coords'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_angles'] = torch.as_tensor(0.).to('cuda')
    if aux_num:
        for i in range(aux_num):
            # dn aux loss
            if training and 'output_known_lbs_polys' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_polys(output_known_coord[i], known_polys, known_lengths )
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_coords'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses

