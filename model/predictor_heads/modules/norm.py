import torch
import torch.nn as nn

class Norm(nn.Module):
    def __init__(self, norm_type, hidden_dim=64):
        super(Norm, self).__init__()
        # assert norm_type in ['bn', 'ln', 'gn', None]
        self.norm = None
        self.nodes_per_img = None
        if norm_type == 'batch':
            self.norm = nn.BatchNorm1d(hidden_dim)
        elif norm_type == 'layer':
            self.norm = nn.LayerNorm1d(hidden_dim)
        elif norm_type == 'instance':
            self.norm = norm_type
            self.mean_scale = 1
        elif 'graph' in norm_type:
            self.norm = norm_type
            self.weight = nn.Parameter(torch.ones(hidden_dim))
            self.bias = nn.Parameter(torch.zeros(hidden_dim))
            self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, tensor, nodes_per_img=None, print_=False):
        if self.norm is not None and type(self.norm) != str:
            return self.norm(tensor)
        elif self.norm is None:
            return tensor

        if nodes_per_img is None:
            nodes_per_img = self.nodes_per_img

        if self.norm != 'graph_batch':
            # compute mean, shift
            batch_size = len(nodes_per_img)
            batch_list = torch.tensor(nodes_per_img).long().to(tensor.device)
            batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
            batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
            mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
            mean = mean.scatter_add_(0, batch_index, tensor)
            mean = (mean.T / (batch_list + 1e-6)).T # add to denom for stability
            mean = mean.repeat_interleave(batch_list, dim=0)
            sub = tensor - mean * self.mean_scale

            # compute std, scale
            std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
            std = std.scatter_add_(0, batch_index, sub.pow(2))
            std = ((std.T / (batch_list + 1e-6)).T + 1e-6).sqrt()
            std = std.repeat_interleave(batch_list, dim=0)

            if self.norm == 'graph':
                norm_result = self.weight * sub / std + self.bias
            else:
                norm_result = sub / std

        else:
            # compute mean, shift
            sub = tensor - tensor.mean(0) * self.mean_scale
            std = tensor.std(0) + 1e-6

            # compute std, scale
            norm_result = self.weight * sub / std + self.bias

        return norm_result

    def train(self, mode=True):
        super(Norm, self).train(mode)

        if self.norm == 'graph':
            self.weight.requires_grad = mode
            self.bias.requires_grad = mode
            self.mean_scale.requires_grad = mode
