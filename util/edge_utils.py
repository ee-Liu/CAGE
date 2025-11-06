import numpy as np
import cv2
from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry import LineString
from shapely.ops import unary_union

def remove_rooms_with_iou(polygon_list):
    # Compute the IOU between each pair
    room_map_list = []
    for room_ind, poly in enumerate(polygon_list):
        room_map = np.zeros((256, 256))
        cv2.fillPoly(room_map, [np.array(poly.exterior.coords, dtype=np.int32)[:-1]], color=1.)
        room_map_list.append(room_map)

    access_mat = np.zeros((len(polygon_list), len(polygon_list)))
    remove_indices = []
    for idx0, polygon0 in enumerate(polygon_list):
        for idx1, polygon1 in enumerate(polygon_list):
            if idx0 == idx1 or access_mat[idx0][idx1] == 1 or access_mat[idx1][idx0] == 1:
                continue
            # compute the iou bewteen polygon0 and polygon1
            intersection = ((room_map_list[idx0] + room_map_list[idx1]) == 2)
            union = ((room_map_list[idx0] + room_map_list[idx1]) >= 1)
            iou = np.sum(intersection) / (np.sum(union) + 1)
            if iou > 0.4:
                remove_indices.append([idx0, idx1])
            access_mat[idx0][idx1] = 1
            access_mat[idx1][idx0] = 1
    
    for remove_index in remove_indices:
        idx0, idx1 = remove_index[0], remove_index[1]
        polygon0, polygon1 = polygon_list[idx0], polygon_list[1]
        poly_area0, poly_area1 = polygon0.area, polygon1.area
        if poly_area0 > poly_area1:
            del polygon_list[idx1]
        else:
            del polygon_list[idx0]
    return polygon_list

def refine_rooms(polygon_list,overlap):
    access_mat = np.zeros((len(polygon_list), len(polygon_list)))
    for idx0, polygon0 in enumerate(polygon_list):
        for idx1, polygon1 in enumerate(polygon_list):
            if idx0 == idx1 or access_mat[idx0][idx1] == 1 or access_mat[idx1][idx0] == 1:
                continue
            if polygon0.intersects(polygon1):
                intersection = polygon0.intersection(polygon1)
                intersection_area = intersection.area
                if intersection_area >= 20:
                    overlap = True
                    # remove intersection from larger polygon
                    area0, area1 = polygon0.area, polygon1.area

                    if area0 > area1 and area1 > intersection_area:
                        polygon0 = polygon0.difference(intersection)
                        polygon1 = polygon1.union(intersection)
                    elif area0 < area1 and area0 > intersection_area:
                        polygon1 = polygon1.difference(intersection)
                        polygon0 = polygon0.union(intersection)
                    elif area0 > area1 and area1 == intersection_area:
                        polygon0 = polygon0.difference(intersection)
                        polygon1 = polygon1.union(intersection)

                        if polygon0.geom_type == 'MultiPolygon':
                            polygon_num = len(polygon0.geoms)
                            max_area = 0
                            smaller_polygon = None
                            for _ in range(polygon_num):
                                area = polygon0.geoms[_].area
                                if area > max_area:
                                    smaller_polygon = polygon0.geoms[_]
                                    max_area = area
                            polygon0 = smaller_polygon
                    elif area0 < area1 and area0 == intersection_area:
                        polygon1 = polygon1.difference(intersection)
                        polygon0 = polygon0.union(intersection)
                        if polygon1.geom_type == 'MultiPolygon':
                            polygon_num = len(polygon1.geoms)
                            max_area = 0
                            smaller_polygon = None
                            for _ in range(polygon_num):
                                area = polygon1.geoms[_].area
                                if area > max_area:
                                    smaller_polygon = polygon1.geoms[_]
                                    max_area = area
                            polygon1 = smaller_polygon
                    polygon_list[idx0] = polygon0
                    polygon_list[idx1] = polygon1
            access_mat[idx0][idx1] = 1
            access_mat[idx1][idx0] = 1
    return polygon_list,overlap

def merge_points(points, threshold=10):
    merged_points = points.copy()

    dp_points = []
    for i in range(merged_points.shape[0]):
        dp_points.append([merged_points[i]])
    dp_points = np.array(dp_points)
    try:
        dp_points = cv2.approxPolyDP(dp_points, epsilon=threshold, closed=True)
        merged_points = []
        for i in range(dp_points.shape[0]):
            merged_points.append(dp_points[i][0])
        merged_points = np.array(merged_points)
        return merged_points
    except:
        return points

def remove_short_edges(edges,pred_logits, threshold=5):

    dx = edges[:, 2] - edges[:, 0]  # x2 - x1
    dy = edges[:, 3] - edges[:, 1]  # y2 - y1
    
    dist_sq = dx ** 2 + dy ** 2
    threshold_sq = threshold ** 2
    mask = dist_sq > threshold_sq
    
    filtered_edges = edges[mask]
    pred_logits = pred_logits[mask]
    return filtered_edges,pred_logits
def is_parallel(vec1: np.ndarray, vec2: np.ndarray, angle_threshold: float = 5.0):

    norm1 = np.linalg.norm(vec1, axis=1, keepdims=True)
    norm2 = np.linalg.norm(vec2, axis=1, keepdims=True)
    unit_vec1 = vec1 / (norm1 + 1e-8)
    unit_vec2 = vec2 / (norm2 + 1e-8)

    cos_theta = np.sum(unit_vec1 * unit_vec2, axis=1)
    theta = np.arccos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi

    return (theta < angle_threshold) | (theta > (180 - angle_threshold))

def point_to_segment_distance(P, A, B):
    P = np.atleast_2d(P)
    A = np.atleast_2d(A)
    B = np.atleast_2d(B)

    AB = B - A
    AP = P - A
    AB_len_sq = np.sum(AB ** 2, axis=1, keepdims=True) + 1e-12
    t = np.sum(AP * AB, axis=1, keepdims=True) / AB_len_sq
    t_clipped = np.clip(t, 0.0, 1.0)
    proj = A + t_clipped * AB
    dist = np.linalg.norm(P - proj, axis=1)
    return dist



def compute_intersections_matrix(edges, threshold=10):
    
    starts = edges[:,:2]
    ends = edges[:,2:]
    n = len(edges)

    next_starts = np.roll(starts, -1, axis=0)
    next_ends = np.roll(ends, -1, axis=0)
    dist_sq = np.sum((ends - next_starts) ** 2, axis=1)
    mask = dist_sq <= threshold ** 2
    if ~mask[-1]:
        mask[-1] = True


    
    A = starts  
    B = ends 
    C = next_starts 
    D = next_ends  


    AB = B - A
    CD = D - C
    AC = C-A
    parallel_mask = is_parallel(AB, CD)


    det = AB[:, 0] * CD[:, 1] - AB[:, 1] * CD[:, 0]
    abs = np.abs(det)
    valid = np.abs(det) > 1e-6


    t = (CD[:, 1] * AC[:, 0] - CD[:, 0] * AC[:, 1]) / (det + 1e-6)
    s = (AB[:, 1] * AC[:, 0] - AB[:, 0] * AC[:, 1]) / (det + 1e-6)

    intersections = A + t[:, None] * AB

    d_to_AB = point_to_segment_distance(intersections, A, B)
    d_to_CD = point_to_segment_distance(intersections, C, D)

    eps_param = 0.1 
    valid_both = valid & (~parallel_mask) & ((((t >= 0 - eps_param) & (t <= 1 + eps_param)) & ((s >= 0 - eps_param) | (s <= 1 + eps_param))))
    valid_first=valid & (t>=0-eps_param) & (t<=1+eps_param) & ((s<0-eps_param)|(s>1+eps_param))&(~parallel_mask)
    valid_second=valid &((t<0-eps_param)|(t>1+eps_param))&(s>=0-eps_param)&(s<=1+eps_param)&(~parallel_mask)
    valid_out=valid &((t<0-eps_param)|(t>1+eps_param))&((s<0-eps_param)|(s>1+eps_param))&(~parallel_mask)


    for i,useful in enumerate(valid_both):
        if useful:
            mask[i] = True
    for i,useful in enumerate(valid_first):
        if useful:
            mask[i] = True
    for i,useful in enumerate(valid_second):
        if useful:
            mask[i] = True
    for i,useful in enumerate(valid_out):
        if useful:
            mask[i] = True


    return mask, intersections, valid, valid_both, valid_first, valid_second, valid_out

def remove_multi_polygon(polygon_lst):
    for poly_idx, polygon in enumerate(polygon_lst):
        connect_edges = []
        if isinstance(polygon, MultiPolygon):
            poly_eqs = []
            poly_pts = []
            poly_v = []

            for sub_polygon in polygon.geoms:

                poly_np = np.array(sub_polygon.exterior.coords, dtype=np.uint8)[:-1]
                simplified_poly_np = simplify_polygon(poly_np)
                poly_np = simplified_poly_np
                poly_eq = []
                poly_pt = []
                for i in range(len(poly_np)-1):
                    start_point = poly_np[i]
                    end_point = poly_np[(i + 1) % len(poly_np)]
                    if start_point[0] == end_point[0]:  # Vertical line
                        line_eq = [float('inf'), start_point[0]]
                    elif start_point[1] == end_point[1]:  # Horizontal line
                        line_eq = [0, start_point[1]]
                    else:
                        line_eq = np.polyfit([start_point[0], end_point[0]], [start_point[1], end_point[1]], 1)
                    poly_eq.append(line_eq)
                    poly_pt.append([start_point, end_point])
                    
                poly_v.append(poly_np)
                poly_eqs.append(poly_eq)
                poly_pts.append(poly_pt)
            
            assert len(poly_v) == len(poly_eqs) == 2
            

            for i, poly_eq in enumerate(poly_eqs):
                src_polygon = poly_v[i]
                tgt_polygon = poly_v[(i+1)%len(poly_v)]
                poly_pt = poly_pts[i]
                for eq_i, eq in enumerate(poly_eq):
                    pt = poly_pt[eq_i]

                    if eq[0] == float('inf'):  # Vertical line
                        line = LineString([(eq[1], 0), (eq[1], 255)])
                    elif eq[0] == 0:  # Horizontal line
                        line = LineString([(0, eq[1]), (255, eq[1])])
                    else:  # Diagonal line
                        line = LineString([(0, eq[1]), (255, eq[0] * 255 + eq[1])])
                    
                    intersection = line.intersection(Polygon(tgt_polygon))

                    # If there is an intersection, return the first intersection point
                    if not intersection.is_empty:
                        if intersection.geom_type == 'Point':
                            intersection_point = (intersection.x, intersection.y)
                        elif intersection.geom_type == 'MultiPoint':
                            intersection_point = (intersection[0].x, intersection[0].y)
                        elif intersection.geom_type == 'LineString':
                            intersection_point = [(intersection.coords[0][0], intersection.coords[0][1]), 
                                                  (intersection.coords[1][0], intersection.coords[1][1])]
                        else:
                            continue

                        # Calculate the distance between points in pt and intersection_point
                        distances = []
                        for p in pt:
                            for ip in intersection_point:
                                distance = np.sqrt((p[0] - ip[0])**2 + (p[1] - ip[1])**2)
                                distances.append((distance, p, ip))

                        # Find the pair with the minimum distance
                        min_distance, min_pt, min_ip = min(distances, key=lambda x: x[0])

                        # Check if the line segment intersects with src_polygon
                        line_segment = LineString([min_pt, min_ip])
                        if line_segment.intersection(Polygon(src_polygon)).geom_type == 'Point':
                            connect_edges.append([min_pt, min_ip])
            
            if len(connect_edges) == 1:
                merge_polygon = polygon[0] if polygon[0].area > polygon[1].area else polygon[1]
            else:
                edge_points = np.array([point for edge in connect_edges for point in edge], dtype=np.uint8)
                new_edge_points = []
                new_edge_points.append(edge_points[0])
                edge_points = np.delete(edge_points, 0, axis=0)
                while edge_points.shape[0] > 0:
                    for idx, point in enumerate(edge_points):
                        if point[0] == new_edge_points[-1][0] or point[1] == new_edge_points[-1][1]:
                            new_edge_points.append(point)
                            edge_points = np.delete(edge_points, idx, axis=0)
                            break
                edge_polygon = Polygon(new_edge_points)                
                merge_polygon = unary_union([Polygon(src_polygon), Polygon(tgt_polygon), edge_polygon])

            poly_np = np.array(merge_polygon.exterior.coords, dtype=np.uint8)[:-1]
            simplified_poly_np = simplify_polygon(poly_np)
            poly_np = np.concatenate([simplified_poly_np, simplified_poly_np[None, 0]])
            update_polygon = Polygon(poly_np)
            polygon_lst[poly_idx] = update_polygon

    for polygon in polygon_lst:
        assert polygon.geom_type == 'Polygon'
    return polygon_lst

def get_corners_from_edges(edges,pred_logits, threshold=10):
    """convert edges to corners"""
    if len(edges) < 3:
        return edges
 
    filtered_edges = edges
    mask, intersections, valid, valid_both, valid_first, valid_second, valid_out= compute_intersections_matrix(filtered_edges,threshold)

    valid = valid[mask]
    intersections = intersections[mask]
    valid_both = valid_both[mask]
    valid_first = valid_first[mask]
    valid_second = valid_second[mask]
    valid_out = valid_out[mask]

    valid_indices = np.where(mask)[0]
    index_map = {orig_idx: arr_idx for arr_idx, orig_idx in enumerate(valid_indices)}

    corners = []
    n=len(filtered_edges)
    for i in range(len(filtered_edges)):
        currrent_start = filtered_edges[i, :2]
        current_end = filtered_edges[i, 2:]
        next_start = filtered_edges[(i + 1) % n, :2]
        next_end = filtered_edges[(i + 1) % n, 2:]

        if mask[i]:
            
            arr_idx = index_map.get(i, -1)
           
            if arr_idx != -1 and valid[arr_idx]:
                dist_to_end = np.linalg.norm(intersections[arr_idx] - current_end)
                dist_to_start = np.linalg.norm(intersections[arr_idx] - next_start)
                dist_corners = np.linalg.norm(current_end - next_start)

                avg_dist_to_corners = (dist_to_end + dist_to_start) / 2
                if valid_both[arr_idx]:
                    corners.append(intersections[arr_idx])

                elif valid_first[arr_idx] or valid_second[arr_idx]:
                    if dist_corners <= avg_dist_to_corners:

                        if not corners or tuple(corners[-1]) != tuple(current_end):
                            corners.append(current_end)
                        if tuple(next_start) != tuple(current_end):
                            corners.append(next_start)

                    else:
                        corners.append(intersections[arr_idx])

                elif valid_out[arr_idx]:
                    if dist_to_end<=2.5 and dist_to_start<=2.5:

                        corners.append(intersections[arr_idx])

                    else:
                        if not corners or tuple(corners[-1]) != tuple(current_end):
                            corners.append(current_end)
                        if tuple(next_start) != tuple(current_end):
                            corners.append(next_start)

                else:
                    if not corners or tuple(corners[-1]) != tuple(current_end):
                        corners.append(current_end)
                    if tuple(next_start) != tuple(current_end):
                        corners.append(next_start)

            else:
                if not corners or tuple(corners[-1]) != tuple(current_end):
                    corners.append(current_end)
                if tuple(next_start) != tuple(current_end):
                    corners.append(next_start)

        else:
            if not corners or tuple(corners[-1]) != tuple(current_end) :
                corners.append(current_end)
            if tuple(next_start) != tuple(current_end):
                corners.append(next_start)


    corners = np.array(corners)

    return corners
def simplify_polygon(input_poly):
    def is_angle_change(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p2
        angle = np.arctan2(v2[1], v2[0]) - np.arctan2(v1[1], v1[0])

        return np.abs(angle) > 1e-2  

    simplified_poly_np = []
    for i in range(len(input_poly)):
        if is_angle_change(input_poly[(i - 1)%len(input_poly)], input_poly[i], input_poly[(i + 1)%len(input_poly)]):
            simplified_poly_np.append(input_poly[i])

    simplified_poly_np = np.array(simplified_poly_np, dtype=np.uint8)
    
    return simplified_poly_np

def remove_duplicate_corners(polygon):
    
    simplified_poly_np = simplify_polygon(polygon)
   

    return simplified_poly_np
