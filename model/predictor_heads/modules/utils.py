import torch
from torch import nn

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
