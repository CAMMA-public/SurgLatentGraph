#!/usr/bin/python
#
# Copyright 2018 Google LLC
# Modification copyright 2020 Helisa Dhamo
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

from .layers import get_normalization_2d, get_activation

from .SPADE.normalization import SPADE
import torch.nn.utils.spectral_norm as spectral_norm

class CRNBlock(nn.Module):
  """
  Cascaded refinement network (CRN) block, as described in:
  Qifeng Chen and Vladlen Koltun,
  "Photographic Image Synthesis with Cascaded Refinement Networks",
  ICCV 2017
  """
  def __init__(self, layout_dim, input_dim, output_dim,
               normalization='instance', activation='leakyrelu'):
    super(CRNBlock, self).__init__()

    layers = []

    layers.append(nn.Conv2d(layout_dim + input_dim, output_dim,
                            kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization))
    layers.append(get_activation(activation))
    layers.append(nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1))
    layers.append(get_normalization_2d(output_dim, normalization))
    layers.append(get_activation(activation))
    layers = [layer for layer in layers if layer is not None]
    for layer in layers:
      if isinstance(layer, nn.Conv2d):
        nn.init.kaiming_normal_(layer.weight)
    self.net = nn.Sequential(*layers)

  def forward(self, layout, feats):
    _, CC, HH, WW = layout.size()
    _, _, H, W = feats.size()
    assert HH >= H
    if HH > H:
      factor = round(HH // H)
      assert HH % factor == 0
      assert WW % factor == 0 and WW // factor == W
      layout = F.avg_pool2d(layout, kernel_size=factor, stride=factor)

    net_input = torch.cat([layout, feats], dim=1)
    out = self.net(net_input)
    return out

class SPADEResnetBlock(nn.Module):
    """
    ResNet block used in SPADE.
    It differs from the ResNet block of pix2pixHD in that
    it takes in the segmentation map as input, learns the skip connection if necessary,
    and applies normalization first and then convolution.
    This architecture seemed like a standard architecture for unconditional or
    class-conditional GAN architecture using residual block.
    The code was inspired from https://github.com/LMescheder/GAN_stability.
    """

    def __init__(self, fin, fout, seg_nc, src_nc, spade_config_str='spadebatch3x3', spectral=True):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)
        self.src_nc = src_nc

        # create conv layers
        self.conv_0 = nn.Conv2d(fin+self.src_nc, fmiddle, kernel_size=3, padding=1)
        self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if spectral:
            self.conv_0 = spectral_norm(self.conv_0)
            self.conv_1 = spectral_norm(self.conv_1)
            if self.learned_shortcut:
                self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        self.norm_0 = SPADE(spade_config_str, fin, seg_nc)
        self.norm_1 = SPADE(spade_config_str, fmiddle, seg_nc)
        if self.learned_shortcut:
            self.norm_s = SPADE(spade_config_str, fin, seg_nc)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, seg_, x):

        seg_ = F.interpolate(seg_, size=x.size()[2:], mode='nearest')

        # only use the layout map as input to SPADE norm (not the source image channels)
        layout_only_dim = seg_.size(1) - self.src_nc
        in_img = seg_[:, layout_only_dim:, :, :]
        seg = seg_[:, :layout_only_dim, :, :]

        x_s = self.shortcut(x, seg)
        dx = torch.cat([self.norm_0(x, seg), in_img],1)
        dx = self.conv_0(self.actvn(dx))
        dx = self.conv_1(self.actvn(self.norm_1(dx, seg)))

        out = x_s + dx

        return out

    def shortcut(self, x, seg):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
