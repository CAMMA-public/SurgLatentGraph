import torch
from torch import nn
import json
import cv2
from pycocotools import mask
import numpy as np
import torch.nn.functional as F

def box_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # calculate top-left and bottom-right
    br = torch.min(boxes1[:, None, :2], boxes2[None, :, :2]) # N x M x 2
    tl = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:]) # N x M x 2

    union_boxes = torch.cat([br, tl], dim=-1) # B x N x M x 4

    return union_boxes

class SelectItem(nn.Module):
    def __init__(self, item_index):
        super(SelectItem, self).__init__()
        self._name = 'selectitem'
        self.item_index = item_index

    def forward(self, inputs):
        return inputs[self.item_index]

class SqueezeItem(nn.Module):
    def __init__(self, squeeze_index):
        super(SqueezeItem, self).__init__()
        self._name = 'squeezeitem'
        self.squeeze_index = squeeze_index

    def forward(self, inputs):
        return inputs.squeeze(self.squeeze_index)

class DuplicateItem(nn.Module):
    def __init__(self):
        super(DuplicateItem, self).__init__()
        self._name = 'duplicateitem'

    def forward(self, inputs):
        return inputs, torch.zeros_like(inputs)

class CustomSequential(nn.Sequential):
    def forward(self, input):
        for module in self._modules.values():
            if isinstance(input, tuple):
                input = module(*input)
            else:
                input = module(input)

        return input

def apply_sparse_mask(mat, mask):
    # get intersection mask
    if mat.ndim > mask.ndim:
        intersect_mask = torch.stack([torch.masked._combine_input_and_mask(
            sum, mask.coalesce(), mat.coalesce()[..., i]) \
                    for i in range(mat.shape[-1])], -1)
    else:
        intersect_mask = torch.masked._combine_input_and_mask(sum, mask.coalesce(),
                mat.coalesce())

    intersect_mask = torch.masked._combine_input_and_mask(sum, intersect_mask.coalesce(),
            intersect_mask.coalesce())

    # mask mat
    mat = torch.masked._combine_input_and_mask(sum, mat.coalesce(),
            intersect_mask.coalesce())

    # remove any excess 0s
    mat = torch.masked._combine_input_and_mask(sum, mat.coalesce(), mat.coalesce())

    return mat

def get_sparse_mask_inds(mat, inds, N):
    sparse_inds = []
    for dim, i in enumerate(inds):
        if i == -1:
            sparse_inds.append(torch.arange(mat.shape[0], device=mat.device).repeat_interleave(N))

        elif isinstance(i, tuple) or isinstance(i, list):
            sparse_inds.append(torch.arange(i[0], i[1], device=mat.device))

        else:
            sparse_inds.append(torch.ones(1, device=mat.device).repeat_interleave(N) * i)

    try:
        sparse_inds = torch.stack(sparse_inds)
    except:
        raise ValueError("Invalid indices for get_sparse_mask_inds!")

    return sparse_inds

def dense_mask_to_polygon_mask(dense_instance_mask, N):
    contours = cv2.findContours(dense_instance_mask.detach().cpu().numpy().astype(np.uint8),
            cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)[0]

    polygon_mask = []
    for contour in contours:
        p_mask = contour.squeeze().ravel().astype(float).tolist()
        polygon_mask.append(p_mask)

    if len(polygon_mask) > 1:
        # select largest polygon
        lengths = [len(p) for p in polygon_mask]
        largest_ind = lengths.index(max(lengths))
        polygon_mask = polygon_mask[largest_ind]

    # downsample to N points
    polygon_mask = torch.tensor(polygon_mask).to(dense_instance_mask.device).view(-1, 2)
    if polygon_mask.shape[0] == 0:
        downsampled_polygon_mask = torch.zeros(N, 2).to(polygon_mask.device)
    else:
        downsampled_polygon_mask = F.interpolate(polygon_mask.T.unsqueeze(-1).unsqueeze(0), size=(N, 1))[0][..., 0].T

    return downsampled_polygon_mask
