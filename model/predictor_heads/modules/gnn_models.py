#!/usr/bin/python
#
# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import EGATConv
from torch_geometric.nn import FastRGCNConv, RGATConv, SAGEConv
import copy
from .layers import build_mlp
from .norm import Norm

def _init_weights(module):
    if hasattr(module, 'weight'):
        if isinstance(module, nn.Linear):
            nn.init.kaiming_normal_(module.weight)

class GraphTripleConv(nn.Module):
    def __init__(self, input_dim_obj, input_dim_pred, output_dim=None, output_dim_pred=None,
            hidden_dim=512, pooling='avg', mlp_normalization='none', skip_connect=False,
            dropout=0.0, use_net2=True, use_edges=True, final_nonlinearity=True, causal=False):
        super(GraphTripleConv, self).__init__()
        if output_dim is None:
            output_dim = input_dim_obj
        if output_dim_pred is None:
            output_dim_pred = input_dim_pred
        self.input_dim_obj = input_dim_obj
        self.input_dim_pred = input_dim_pred
        self.output_dim = output_dim
        self.output_dim_pred = output_dim_pred
        self.hidden_dim = hidden_dim
        self.skip_connect = skip_connect
        self.use_net2 = use_net2
        self.use_edges = use_edges
        self.final_nonlinearity = final_nonlinearity
        self.causal = causal
        if mlp_normalization is not None and mlp_normalization != 'none':
            self.norm = Norm(mlp_normalization, output_dim)
        else:
            self.norm = None

        assert pooling in ['sum', 'avg'], 'Invalid pooling "%s"' % pooling

        self.pooling = pooling
        if self.use_net2:
            if self.use_edges:
                net1_layers = [2 * input_dim_obj + input_dim_pred, hidden_dim,
                        2 * hidden_dim + output_dim_pred]
            else:
                net1_layers = [2 * input_dim_obj, hidden_dim, 2 * hidden_dim]

            net2_layers = [hidden_dim, hidden_dim, output_dim]
        else:
            if self.use_edges:
                net1_layers = [2 * input_dim_obj + input_dim_pred, 2 * output_dim + output_dim_pred]
            else:
                net1_layers = [2 * input_dim_obj, hidden_dim, 2 * output_dim]

            net2_layers = []

        self.net1 = build_mlp(net1_layers, dropout=dropout, final_nonlinearity=self.use_net2)
        self.net1.apply(_init_weights)

        if self.use_net2:
            self.net2 = build_mlp(net2_layers, dropout=dropout, final_nonlinearity=False)
            self.net2.apply(_init_weights)

        if self.skip_connect:
            assert self.input_dim_obj == self.output_dim
            self.skip_projector = torch.nn.Identity()
            #self.skip_projector = torch.nn.Linear(self.input_dim_obj, self.output_dim)

    def forward(self, obj_vecs, pred_vecs, edges, nodes_per_img):
        """
        Inputs:
        - obj_vecs: FloatTensor of shape (num_objs, D) giving vectors for all objects
        - pred_vecs: FloatTensor of shape (num_triples, D) giving vectors for all predicates
        - edges: LongTensor of shape (num_triples, 2) where edges[k] = [i, j] indicates the
          presence of a triple [obj_vecs[i], pred_vecs[k], obj_vecs[j]]

        Outputs:
        - new_obj_vecs: FloatTensor of shape (num_objs, D) giving new vectors for objects
        - new_pred_vecs: FloatTensor of shape (num_triples, D) giving new vectors for predicates
        """
        dtype, device = obj_vecs.dtype, obj_vecs.device
        num_objs, num_triples = obj_vecs.size(0), pred_vecs.size(0)
        Din_obj, Din_pred, H, Dout, Dout_pred = self.input_dim_obj, self.input_dim_pred, \
                self.hidden_dim, self.output_dim, self.output_dim_pred

        # Break apart indices for subjects and objects; these have shape (num_triples,)
        s_idx = edges[:, 0].contiguous()
        o_idx = edges[:, 1].contiguous()

        # Get current vectors for subjects and objects; these have shape (num_triples, Din)
        cur_s_vecs = obj_vecs[s_idx]
        cur_o_vecs = obj_vecs[o_idx]

        # Get current vectors for triples; shape is (num_triples, 3 * Din)
        # Pass through net1 to get new triple vecs; shape is (num_triples, 2 * H + Dout_pred)
        if self.use_edges:
            cur_t_vecs = torch.cat([cur_s_vecs, pred_vecs, cur_o_vecs], dim=1)
        else:
            cur_t_vecs = torch.cat([cur_s_vecs, cur_o_vecs], dim=1)

        new_t_vecs = self.net1(cur_t_vecs)

        # Break apart into new s, p, and o vecs; s and o vecs have shape (num_triples, H) and
        # p vecs have shape (num_triples, Dout_pred)
        if self.use_net2:
            new_s_vecs = new_t_vecs[:, :H]
            if self.use_edges:
                new_o_vecs = new_t_vecs[:, (H+Dout_pred):(2 * H + Dout_pred)]
                new_p_vecs = new_t_vecs[:, H:(H+Dout_pred)]
            else:
                new_o_vecs = new_t_vecs[:, H:]
                new_p_vecs = pred_vecs

            # Allocate space for pooled object vectors of shape (num_objs, H)
            pooled_obj_vecs = torch.zeros(num_objs, H, dtype=dtype, device=device)

        else:
            new_s_vecs = new_t_vecs[:, :Dout]
            if self.use_edges:
                new_p_vecs = new_t_vecs[:, Dout:(Dout+Dout_pred)]
                new_o_vecs = new_t_vecs[:, (Dout+Dout_pred):]
            else:
                new_o_vecs = new_t_vecs[:, Dout:]
                new_p_vecs = pred_vecs

            # Allocate space for pooled object vectors of shape (num_objs, H)
            pooled_obj_vecs = torch.zeros(num_objs, Dout, dtype=dtype, device=device)

        # Use scatter_add to sum vectors for objects that appear in multiple triples;
        # we first need to expand the indices to have shape (num_triples, D)
        s_idx_exp = s_idx.view(-1, 1).expand_as(new_s_vecs)
        o_idx_exp = o_idx.view(-1, 1).expand_as(new_o_vecs)

        # In causal mode, we only update the node that the edge points to, assuming that
        # the graph has been constructed such that future nodes do not have edges to
        # past nodes (e.g. not (t+n, t), only (t, t+n)).
        if not self.causal:
            pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, s_idx_exp, new_s_vecs)

        pooled_obj_vecs = pooled_obj_vecs.scatter_add(0, o_idx_exp, new_o_vecs)

        if self.pooling == 'avg':
            #print("here i am, would you send me an angel")
            # Figure out how many times each object has appeared, again using
            # some scatter_add trickery.
            obj_counts = torch.zeros(num_objs, dtype=dtype, device=device)
            ones = torch.ones(num_triples, dtype=dtype, device=device)
            obj_counts = obj_counts.scatter_add(0, s_idx, ones)
            obj_counts = obj_counts.scatter_add(0, o_idx, ones)

            # Divide the new object vectors by the number of times they
            # appeared, but first clamp at 1 to avoid dividing by zero;
            # objects that appear in no triples will have output vector 0
            # so this will not affect them.
            obj_counts = obj_counts.clamp(min=1)
            pooled_obj_vecs = pooled_obj_vecs / obj_counts.view(-1, 1)
        # Send pooled object vectors through net2 to get output object vectors,
        # of shape (num_objs, Dout)
        if self.use_net2:
            new_obj_vecs = self.net2(pooled_obj_vecs)
        else:
            new_obj_vecs = pooled_obj_vecs

        # apply norm
        if self.norm is not None:
            new_obj_vecs = self.norm(new_obj_vecs, nodes_per_img)

        # apply ReLU
        if self.final_nonlinearity:
            new_obj_vecs = F.relu(new_obj_vecs)
            if not self.use_net2 and self.use_edges:
                new_p_vecs = F.relu(new_p_vecs)

        if self.skip_connect:
            new_obj_vecs = new_obj_vecs + self.skip_projector(obj_vecs)
            if self.use_edges:
                new_p_vecs = new_p_vecs + pred_vecs

        return new_obj_vecs, new_p_vecs

class GraphTripleConvNet(nn.Module):
    def __init__(self, input_dim_obj, input_dim_pred, output_dim=None, output_dim_pred=None,
            num_layers=5, hidden_dim=512, pooling='avg', mlp_normalization='none', skip_connect=False,
            dropout=0.0, use_net2=True, use_edges=True, final_nonlinearity=True, causal=False):
        super(GraphTripleConvNet, self).__init__()

        self.num_layers = num_layers
        self.gconvs = nn.ModuleList()
        gconv_kwargs = {
          'input_dim_obj': input_dim_obj,
          'input_dim_pred': input_dim_pred,
          'output_dim': output_dim,
          'output_dim_pred': output_dim_pred,
          'hidden_dim': hidden_dim,
          'pooling': pooling,
          'mlp_normalization': mlp_normalization,
          'skip_connect': skip_connect,
          'dropout': dropout,
          'use_net2': use_net2,
          'use_edges': use_edges,
          'causal': causal,
        }

        # modify input dim after first conv layer
        self.gconvs.append(GraphTripleConv(**gconv_kwargs))
        if output_dim is not None:
            gconv_kwargs['input_dim_obj'] = output_dim

        for _ in range(1, self.num_layers):
            if _ == self.num_layers - 1:
                gconv_kwargs['final_nonlinearity'] = final_nonlinearity

            self.gconvs.append(GraphTripleConv(**gconv_kwargs))

    def forward(self, node_features, edge_features, edge_flats, graph=None):
        if graph is not None:
            nodes_per_img = graph.batch_num_nodes().tolist()
        else:
            nodes_per_img = None

        for i in range(self.num_layers):
            node_features, edge_features = self.gconvs[i](node_features, edge_features, edge_flats, nodes_per_img)

        return node_features, edge_features

class GNN(nn.Module):
    def __init__(self, node_feat_size, edge_feat_size, hidden_dim_size, num_heads,
            num_layers, act=None, norm=None, dropout=0.0, arch='rgcn', num_relations=4,
            use_edges_gcn=False):
        super(GNN, self).__init__()

        # define vars
        self.node_feat_size = node_feat_size
        self.edge_feat_size = edge_feat_size
        self.num_relations = num_relations
        self.arch = arch
        self.use_edges_gcn = use_edges_gcn

        # resolve activation
        if act == 'relu':
            self.act = torch.nn.ReLU()
        elif act == 'leaky-relu':
            self.act = torch.nn.LeakyReLU()

        # edit hidden_dim_size
        out_size = hidden_dim_size // num_heads

        if self.arch == 'egat':
            # define first layer
            self.layers = [EGATConv(self.node_feat_size, self.edge_feat_size, out_size,
                    out_size, num_heads)]

            # for remaining layers, input and output both have hidden_dim_size
            for i in range(num_layers - 1):
                self.layers.append(EGATConv(hidden_dim_size, hidden_dim_size, out_size,
                    out_size, num_heads))

        elif self.arch == 'rgcn':
            # define first layer
            self.layers = [FastRGCNConv(self.node_feat_size, hidden_dim_size,
                num_relations)]

            # for remaining layers, input and output both have hidden_dim_size
            for i in range(num_layers - 1):
                self.layers.append(FastRGCNConv(hidden_dim_size, hidden_dim_size, num_relations))

        elif self.arch == 'rgat':
            if self.use_edges_gcn:
                self.layers = [RGATConv(self.node_feat_size, out_size, num_relations,
                    heads=num_heads, edge_dim=self.edge_feat_size, dropout=dropout)]

                # for remaining layers, input and output both have hidden_dim_size
                for i in range(num_layers - 1):
                    self.layers.append(RGATConv(hidden_dim_size, out_size, num_relations,
                        heads=num_heads, edge_dim=self.edge_feat_size, dropout=dropout))

            else:
                self.layers = [RGATConv(self.node_feat_size, out_size, num_relations,
                    heads=num_heads, dropout=dropout)]

                # for remaining layers, input and output both have hidden_dim_size
                for i in range(num_layers - 1):
                    self.layers.append(RGATConv(hidden_dim_size, out_size, num_relations,
                        heads=num_heads, dropout=dropout))

        else:
            raise ValueError(arch + " is not supported.")

        if norm is not None:
            self.norms = []
            for _ in range(num_layers):
                node_norm_layer = norm(hidden_dim_size)
                edge_norm_layer = norm(hidden_dim_size)

                # node and edge norm
                self.norms.append([copy.deepcopy(node_norm_layer),
                    copy.deepcopy(edge_norm_layer)])

        else:
            self.norms = norm

        if dropout > 0 and self.arch != 'rgat':
            self.dropout = torch.nn.Dropout(dropout)
        else:
            self.dropout = None

    def forward(self, node_features, edge_features, edge_flats, graph):
        for ind, l in enumerate(self.layers):
            update_edge_feats = False
            if self.arch == 'egat':
                node_features, edge_features = l(graph, node_features, edge_features)
                update_edge_feats = True
            elif self.arch == 'rgcn':
                if edge_flats.shape[0] != 2:
                    edge_flats = edge_flats.T
                edge_labels = edge_features[:, :-1 * self.num_relations].argmax(1).long()
                node_features = l(node_features, edge_flats, edge_labels)
            elif self.arch == 'rgat':
                if edge_flats.shape[0] != 2:
                    edge_flats = edge_flats.T
                edge_labels = edge_features[:, :-1 * self.num_relations].argmax(1).long()
                if self.use_edges_gcn:
                    node_features = l(node_features, edge_flats, edge_labels,
                            edge_attr=edge_features)
                else:
                    node_features = l(node_features, edge_flats, edge_labels)

            if node_features.dim() > 2:
                node_features = node_features.flatten(start_dim=1)
            if edge_features is not None and edge_features.dim() > 2:
                edge_features = edge_features.flatten(start_dim=1)

            if self.norms is not None:
                node_features = self.norms[ind][0](node_features)
                if update_edge_feats:
                    edge_features = self.norms[ind][1](edge_features) if edge_features is not None else None

            # activation is done in conv for regular gat
            if self.act is not None:
                node_features = self.act(node_features)
                if update_edge_feats:
                    edge_features = self.act(edge_features) if edge_features is not None else None

            if self.dropout is not None:
                node_features = self.dropout(node_features)
                if update_edge_feats:
                    edge_features = self.dropout(edge_features) if edge_features is not None else None

        return node_features, edge_features

    def to(self, device):
        super().to(device)
        self.layers = [l.to(device) for l in self.layers]

        if self.norms is not None:
            self.norms = [[l.to(device) for l in n] for n in self.norms]

    def train(self, mode):
        super().train(mode)

        self.layers = [l.train(mode) for l in self.layers]
        if self.norms is not None:
            self.norms = [[l.train(mode) for l in n] for n in self.norms]

    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
        for norm in self.norms or []:
            norm.reset_parameters()

class GNNBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.LayerNorm(in_channels, elementwise_affine=True)
        self.conv = SAGEConv(in_channels, out_channels, project=True)

    def reset_parameters(self):
        self.norm.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, edge_index, dropout_mask=None):
        x = self.norm(x).relu()
        if self.training and dropout_mask is not None:
            x = x * dropout_mask
        return self.conv(x, edge_index)

class RevGNN(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout, num_groups=2):
        super().__init__()

        self.dropout = dropout

        self.lin1 = nn.Linear(in_channels, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, out_channels)
        self.norm = nn.LayerNorm(hidden_channels, elementwise_affine=True)

        assert hidden_channels % num_groups == 0
        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = GNNBlock(
                hidden_channels // num_groups,
                hidden_channels // num_groups,
            )
            self.convs.append(GroupAddRev(conv, num_groups=num_groups))

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.norm.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, edge_index):
        x = self.lin1(x)

        # Generate a dropout mask which will be shared across GNN blocks:
        mask = None
        if self.training and self.dropout > 0:
            mask = torch.zeros_like(x).bernoulli_(1 - self.dropout)
            mask = mask.requires_grad_(False)
            mask = mask / (1 - self.dropout)

        for conv in self.convs:
            x = conv(x, edge_index, mask)
        x = self.norm(x).relu()
        x = F.dropout(x, p=self.dropout, training=self.training)
        return self.lin2(x)
