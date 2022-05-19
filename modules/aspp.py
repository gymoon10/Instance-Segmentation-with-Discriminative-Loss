import torch
import torch.nn as nn
import torch.nn.functional as F


class ASPP(nn.Module):
    def __init__(self, sem_classes, ins_classes):
        super(ASPP, self).__init__()

        # 1. 1x1 conv: part(a) in fig.5 of paper
        self.conv_1x1_1 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        # 2. parallel atrous conv1 (out_ch=256): part(a) in fig.5 of paper
        self.conv_3x3_1 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        # 2. parallel atrous conv2
        self.conv_3x3_2 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        # 2. parallel atrous conv3
        self.conv_3x3_3 = nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        # ASPP-Pooling
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_1x1_2 = nn.Conv2d(512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        # 1x1 conv for concatnated parallel modules
        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        # Semantic-egmentation head
        self.conv_1x1_4 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_1x1_5 = nn.Conv2d(128, 2, kernel_size=1)

        # Instance-segmentation head
        self.conv_1x1_4_1 = nn.Conv2d(256, 128, kernel_size=1)
        self.conv_1x1_5_1 = nn.Conv2d(128, 32, kernel_size=1)

    def forward(self, feature_map):
        '''(N, 512, h/16, w/16) for ResNet18_OS16 or ResNet34_OS16
           (N, 512, h/8, w/8) for ResNet18_OS8 or ResNet34_OS8
           (N, 4*512, h/16, w/16) for ResNet50-152'''

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        # parallel modules include atrous conv
        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))  # (N, 256, h/16, w/16)
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))  # (N, 256, h/16, w/16)
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))  # (N, 256, h/16, w/16)
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))  # (N, 256, h/16, w/16)

        # Image Pooling (augment ASPP with image-level feature)
        out_img = self.avg_pool(feature_map)  # (N, 512, 1, 1)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (N, 512, 1, 1)
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")  # (N, 256, h/16, w/16)

        # Concat
        x_dec = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)  # (N, 256*5, h/16, w/16)
        x_dec = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(x_dec)))  # (N, 256, h/16, w/16)

        # Semantic-Seg
        x_dec_sem = self.conv_1x1_4(x_dec)  # (shape: (batch_size, num_classes, h/16, w/16))
        sem_out = self.conv_1x1_5(x_dec_sem)

        # Instance-Seg
        x_dec_ins = self.conv_1x1_4(x_dec)  # (shape: (batch_size, num_classes, h/16, w/16))
        ins_out = self.conv_1x1_5(x_dec_ins)

        return sem_out, ins_out


class ASPP_Bottleneck(nn.Module):
    def __init__(self, num_classes):
        super(ASPP_Bottleneck, self).__init__()

        self.conv_1x1_1 = nn.Conv2d(4 * 512, 256, kernel_size=1)
        self.bn_conv_1x1_1 = nn.BatchNorm2d(256)

        self.conv_3x3_1 = nn.Conv2d(4 * 512, 256, kernel_size=3, stride=1, padding=6, dilation=6)
        self.bn_conv_3x3_1 = nn.BatchNorm2d(256)

        self.conv_3x3_2 = nn.Conv2d(4 * 512, 256, kernel_size=3, stride=1, padding=12, dilation=12)
        self.bn_conv_3x3_2 = nn.BatchNorm2d(256)

        self.conv_3x3_3 = nn.Conv2d(4 * 512, 256, kernel_size=3, stride=1, padding=18, dilation=18)
        self.bn_conv_3x3_3 = nn.BatchNorm2d(256)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.conv_1x1_2 = nn.Conv2d(4 * 512, 256, kernel_size=1)
        self.bn_conv_1x1_2 = nn.BatchNorm2d(256)

        self.conv_1x1_3 = nn.Conv2d(1280, 256, kernel_size=1)  # (1280 = 5*256)
        self.bn_conv_1x1_3 = nn.BatchNorm2d(256)

        self.conv_1x1_4 = nn.Conv2d(256, num_classes, kernel_size=1)

    def forward(self, feature_map):  # (N, 4*512, h/16, w/16))

        feature_map_h = feature_map.size()[2]  # (== h/16)
        feature_map_w = feature_map.size()[3]  # (== w/16)

        out_1x1 = F.relu(self.bn_conv_1x1_1(self.conv_1x1_1(feature_map)))  # (N, 256, h/16, w/16)
        out_3x3_1 = F.relu(self.bn_conv_3x3_1(self.conv_3x3_1(feature_map)))  # (N, 256, h/16, w/16)
        out_3x3_2 = F.relu(self.bn_conv_3x3_2(self.conv_3x3_2(feature_map)))  # (N, 256, h/16, w/16)
        out_3x3_3 = F.relu(self.bn_conv_3x3_3(self.conv_3x3_3(feature_map)))  # (N, 256, h/16, w/16)

        out_img = self.avg_pool(feature_map)  # (N, 512, 1, 1)
        out_img = F.relu(self.bn_conv_1x1_2(self.conv_1x1_2(out_img)))  # (N, 256, 1, 1)
        out_img = F.upsample(out_img, size=(feature_map_h, feature_map_w), mode="bilinear")  # (N, 256, h/16, w/16)

        out = torch.cat([out_1x1, out_3x3_1, out_3x3_2, out_3x3_3, out_img], 1)  # (N, 1280, h/16, w/16)
        out = F.relu(self.bn_conv_1x1_3(self.conv_1x1_3(out)))  # (N, 256, h/16, w/16)

        # Semantic-Seg
        out = self.conv_1x1_4(out)  # (Ne, num_classes, h/16, w/16)

        return out