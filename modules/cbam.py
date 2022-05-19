#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import math
import torch.nn as nn
import torch.nn.functional as F


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0,
                 dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
    
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

    
class ChannelGate(nn.Module):
    '''Generate 2 different(avg, pool) spatial context descriptors to refine input feature'''
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max']):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels

        # Shared MLP
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),  # reduction
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)  # restoration
            )
        
        self.pool_types = pool_types

    def forward(self, x):
        '''x: Input feature  (N, C, h, w)
           kernel_size of pooling operation = (h, w) -> to squeeze the spatial dimension'''
        channel_att_sum = None  # It should be MLP(AvgPool(x)) + MLP(MaxPool(x))
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (N, C, 1, 1)
                channel_att_raw = self.mlp(avg_pool)  # (N, C)

            elif pool_type=='max':
                max_pool = F.max_pool2d(x, (x.size(2), x.size(3)), stride=(x.size(2), x.size(3)))  # (N, C, 1, 1)
                channel_att_raw = self.mlp(max_pool)  # (N, C)

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw

            else:
                channel_att_sum = channel_att_sum + channel_att_raw  # (N, C) - Channel Attention Map
        
        # Sigmoid & Broad-casting (N, C) -> (N, C, 1) -> (N, C, 1, 1) -> (N, C, h, w)
        scale = torch.sigmoid(channel_att_sum).unsqueeze(2).unsqueeze(3).expand_as(x)
        
        # Feature Refinement
        return x * scale  # (N, C, h, w)
    
    
class ChannelPool(nn.Module):
    '''Apply max pooling & avg pooling along the channel axis and concatenate them
       to generate an efficient feature descriptor'''
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)

    
class SpatialGate(nn.Module):
    '''Produce 2D spatial attention map to refine channel-refined feature (sequential)'''
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)

    def forward(self, x):
        '''x: channel-refined feature (sequential)'''
        x_compress = self.compress(x)  # (N, 2, h, w)
        x_out = self.spatial(x_compress)  # (N, 1, h, w) - Spatial Attention Map
        scale = torch.sigmoid(x_out)  # broadcasting

        return x * scale
    

class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial = no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()

    def forward(self, x):
        '''x: Input feature (N, C, h, w)
           x_out: (N, c, h, w)'''
        x_out = self.ChannelGate(x)  # Channel-refinement
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)  # Spatial-refinement

        return x_out  # Refined feature

