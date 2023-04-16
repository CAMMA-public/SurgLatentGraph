import torch

def box_union(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    # calculate top-left and bottom-right
    br = torch.min(boxes1[:, None, :2], boxes2[None, :, :2]) # N x M x 2
    tl = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:]) # N x M x 2

    union_boxes = torch.cat([br, tl], dim=-1) # B x N x M x 4

    return union_boxes
