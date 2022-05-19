import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from modules.instance_counter import InstanceCounter

def make_layer(block, in_channels, channels, num_blocks, stride=1, dilation=1):
    strides = [stride] + [1] * (num_blocks - 1)  # (stride=2, num_blocks=4 --> strides == [2, 1, 1, 1])

    blocks = []
    for stride in strides:
        blocks.append(block(in_channels=in_channels, channels=channels, stride=stride, dilation=dilation))
        in_channels = block.expansion * channels

    layer = nn.Sequential(*blocks)  # (*blocks: call with unpacked list entires as arguments)

    return layer


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, dilation=1):
        super(BasicBlock, self).__init__()

        out_channels = self.expansion * channels

        self.conv1 = nn.Conv2d(in_channels, channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=dilation, dilation=dilation,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

        if (stride != 1) or (in_channels != out_channels):
            conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            bn = nn.BatchNorm2d(out_channels)
            self.downsample = nn.Sequential(conv, bn)

        else:
            self.downsample = nn.Sequential()

    def forward(self, x):  # x: (N, in_ch, h, w)

        out = F.relu(self.bn1(self.conv1(x)))  # (N, channels, h, w) for stride=1 & (N, channels, h/2, w/2) for stride=2
        out = self.bn2(self.conv2(out))  # (N, channels, h, w) for stride=1 & (N, channels, h/2, w/2) for stride=2

        out = out + self.downsample(x)  # (N, channels, h, w) for stride=1 & (N, channels, h/2, w/2) for stride=2

        out = F.relu(out)

        return out


class ResNet34(nn.Module):
    def __init__(self):
        super(ResNet34, self).__init__()

        # pre-trained resnet
        resnet = models.resnet34(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-3])  # only use ~layer4

        # additional 3x3 atrous conv with rate=2
        self.atrous2 = make_layer(BasicBlock, in_channels=256, channels=512, num_blocks=3, stride=1, dilation=2)

    def forward(self, x):
        feature = self.resnet(x)  # (N, 256, h/16, w/16) (it's called c4 since 16 == 2^4)
        output = self.atrous2(feature)  # (N, 512, h/16, w/16)

        return output


from functools import partial
from typing import Any, List, Optional

import torch
from torch import nn
from torch.nn import functional as F


class ASPPConv(nn.Sequential):
    '''3x3 atrous conv with dilation rate
       + B.N & ReLU'''

    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        '''Input: feature map of ResNet (N, 512, h/16, w/16)'''
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),  # (N, 512, 1, 1)
            nn.Conv2d(in_channels, out_channels, 1, bias=False),  # (N, 256, 1, 1)
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        size = x.shape[-2:]  # h/16, w/16
        x = super(ASPPPooling, self).forward(x)

        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)  # (N, 256, h/16, w/16)


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        '''Input: feature map of ResNet (N, 512, h/16, w/16)'''
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []

        # 1x1 conv: part(a) in fig.5 of paper
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)))  # (N, 256, h/16, w/16)

        # parallel atrous convs (out_ch=256): part(a) in fig.5 of paper
        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))  # (N, 256, h/16, w/16)
        modules.append(ASPPConv(in_channels, out_channels, rate2))  # (N, 256, h/16, w/16)
        modules.append(ASPPConv(in_channels, out_channels, rate3))  # (N, 256, h/16, w/16)

        # image pooling: part(b) in fig.5 of paper (augment ASPP with image-level feature)
        modules.append(ASPPPooling(in_channels, out_channels))  # (N, 256, h/16, w/16)

        self.convs = nn.ModuleList(modules)

        # 1x1 conv for concatenated (a), (b)
        self.project = nn.Sequential(
            # 5: # of modules to be concatenated
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1), )

    def forward(self, x):
        # for parallel atorus convs and image pooling
        res = []
        for conv in self.convs:
            res.append(conv(x))

        # concat
        res = torch.cat(res, dim=1)  # (N, 1280=256*5, h/16, w/16)

        return self.project(res)  # (N, 256, h/16, w/16)


class DeepLabHead(nn.Module):
    '''ASPP module + Segmentation head'''

    def __init__(self, in_channels, sem_classes, ins_classes, aspp_dilate=[6, 12, 18]):
        super(DeepLabHead, self).__init__()

        self.in_channels = in_channels
        self.sem_classes = sem_classes
        self.ins_classes = ins_classes

        # ASPP with Semantic-head
        self.sem_classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),  # (N, 256, h/16, w/16)
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, sem_classes, 1)
        )

        # ASPP with Instance-head
        self.ins_classifier = nn.Sequential(
            ASPP(in_channels, aspp_dilate),  # (N, 256, h/16, w/16)
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, ins_classes, 1)
        )

        self._init_weight()

    def forward(self, feature):
        '''feature: output feature map of ResNet34'''
        sem_out = self.sem_classifier(feature)
        ins_out = self.ins_classifier(feature)

        h = feature.shape[-2] * 16;
        w = feature.shape[-1] * 16

        sem_seg_out = F.interpolate(sem_out, size=(h, w), mode="bilinear")  # to recover resolution
        ins_seg_out = F.interpolate(ins_out, size=(h, w), mode="bilinear")  # to recover resolution

        return sem_seg_out, ins_seg_out

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from collections import OrderedDict


class DeepLabV3(nn.Module):
    def __init__(self, backbone=ResNet34(), head=DeepLabHead(512, 2, 32, aspp_dilate=[6, 12, 18])):
        super(DeepLabV3, self).__init__()
        self.backbone = backbone  # ResNet34
        self.head = head

        self.ins_cls_cnn = InstanceCounter(512, use_coordinates=True,
                                           usegpu=True)

    def forward(self, x):
        input_shape = x.shape[-2:]
        features = self.backbone(x)  # (N, 512, h/16, w/16)
        sem_seg_out, ins_seg_out = self.head(features)

        ins_cnt = self.ins_cls_cnn(features)

        return sem_seg_out, ins_seg_out, ins_cnt


#x = torch.randn([2, 3, 256, 256]).cuda()
#model = DeepLabV3().cuda()

#sem, ins, cnt = model(x)
#print(sem.shape)
#print(ins.shape)
#print(cnt)
