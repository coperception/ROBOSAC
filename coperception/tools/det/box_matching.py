import imp
import torch
import numpy as np
from scipy.optimize import linear_sum_assignment
from shapely.geometry import Polygon
"""
Source Code is adapted from:
github.com/matterport/MaskRCNN
YOLOv5/utils/metrics
"""


def convert_format(boxes_array):
    """
    :param array: an array of shape [# bboxs, 8]
    :return: a shapely.geometry.Polygon object
    """

    polygons = [
        Polygon([(box[i*2], box[i*2+1]) for i in range(4)]) for box in boxes_array
    ]
    return np.array(polygons)


def compute_overlaps(boxes1, boxes2):
    """Computes IoU overlaps between two sets of boxes.
    boxes1, boxes2: a np array of boxes
    For better performance, pass the largest set first and the smaller second.
    :return: a matrix of overlaps [boxes1 count, boxes2 count]
    """
    # Compute overlaps to generate matrix [boxes1 count, boxes2 count]
    # Each cell contains the IoU value.

    boxes1 = convert_format(boxes1)
    boxes2 = convert_format(boxes2)
    overlaps = np.zeros((len(boxes1), len(boxes2)))
    for i in range(overlaps.shape[1]):
        box2 = boxes2[i]
        overlaps[:, i] = compute_iou(box2, boxes1)
    return overlaps

def compute_iou(box, boxes):
    """Calculates IoU of the given box with the array of the given boxes.
    box: a polygon
    boxes: a vector of polygons
    Note: the areas are passed in rather than calculated here for
    efficiency. Calculate once in the caller to avoid duplicate work.
    """
    # Calculate intersection areas
    iou = [box.intersection(b).area / box.union(b).area for b in boxes]

    return np.array(iou, dtype=np.float32)



# Hungarian Matching
def linear_assignment(cost_matrix):
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))
 
 
def associate_2_detections(detections1, detections2, iou_threshold=0.5):
    # Boxes assigned by Hungarian Matching is considered a match, contribute 1 intersect item.
    # This function returns the "IoU" of two bbox sets.

    # if detections2 is empty，directly return 0 associations.
    if len(detections2) == 0:
        return 0
 
    iou_matrix = compute_overlaps(detections1, detections2)
    # [[0.73691421 0.         0.         0.        ]
    #  [0.         0.89356082 0.         0.        ]
    #  [0.         0.         0.76781823 0.        ]]
 
    if min(iou_matrix.shape) > 0:
 
        a = (iou_matrix > iou_threshold).astype(np.int32)
        # [[1 0 0 0]
        #  [0 1 0 0]
        #  [0 0 1 0]]
 
        # print(a.sum(1)): [1 1 1]
        # print(a.sum(0)): [1 1 1 0]
 
        # if box with IoU > 0.5 has one-one matching，straight return the result. Or use hungarian matching
        if a.sum(1).max() == 1 and a.sum(0).max() == 1:
 
            matched_indices = np.stack(np.where(a), axis=1)
            # [[0 0]
            #  [1 1]
            #  [2 2]]
        else:
            matched_indices = linear_assignment(-iou_matrix)
    else:
        matched_indices = np.empty(shape=(0, 2))
 
    unmatched_detections1 = []
    for d, det in enumerate(detections1):
        if d not in matched_indices[:, 0]:
            unmatched_detections1.append(d)
 
    unmatched_detections2 = []
    for t, det in enumerate(detections2):
        if t not in matched_indices[:, 1]:
            unmatched_detections2.append(t)
 
    # print(unmatched_detections1) : []
    # print(unmatched_detections2) : [3]
 
    # if matches after hungarian matching has low IoU, also consider them as mis-match.
    matches = []
    for m in matched_indices:
        if iou_matrix[m[0], m[1]] < iou_threshold:
            unmatched_detections1.append(m[0])
            unmatched_detections2.append(m[1])
        else:
            matches.append(m.reshape(1, 2))
 
    # print(matches): [[0 0] [1 1] [2 2]]
    # print(np.array(unmatched_detections1)): []
    # print(np.array(unmatched_detections2)): [3]
    intersect = len(matches)
    union = len(detections1) + len(detections2) - intersect
    jaccard_index = intersect / union
 
    return jaccard_index

