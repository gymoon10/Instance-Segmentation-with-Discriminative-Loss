import torch
from torch import nn
import torch.nn.functional as F

from modules.instance_counter import InstanceCounter
from modules.cbam import CBAM


class double_conv(nn.Module):
    '''(Conv + B.N + ReLU) x 2 with Attention'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()

        # Conv block
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

        # Attention module
        self.cbam = CBAM(out_ch)

    def forward(self, input):
        input = self.conv(input)
        output = self.cbam(input)  # feature-refinement

        return output


class inconv(nn.Module):
    '''Initial block of UNet with Attention'''
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)

        return x


class down(nn.Module):
    '''Down-sampling block with Attention'''
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)

        return x


class up(nn.Module):
    '''Up-sampling block with Attention'''
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

        # bilinear=True -> up-sample with bilinear interpolation (rule-based)
        # bilinear=False -> up-sample with conv (trained, requires sufficient memory)
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        else:
            self.up = nn.ConvTranspose2d(in_ch, out_ch, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        '''x1 should be up-sampled'''
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)

        return x


class outconv(nn.Module):
    '''Last block of UNet with Attention'''
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        # self.conv = nn.Conv2d(in_ch, out_ch, 1)

        #self.cbam = CBAM(in_ch)

        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, in_ch // 2, 1),
            nn.BatchNorm2d(in_ch // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch // 2, out_ch, 1),
        )


    def forward(self, x):
        #x = self.cbam(x)
        x = self.conv(x)

        return x


class UNet_CBAM(nn.Module):
    def __init__(self):
        super(UNet_CBAM, self).__init__()
        self.inc = inconv(3, 64)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 512)

        self.up1 = up(1024, 256)
        self.up2 = up(512, 128)
        self.up3 = up(256, 64)
        self.up4 = up(128, 64)

        # Segmentation heads (parallel)
        self.sem_out = outconv(64, 2)  # semantic classes=2 (f.g / b.g)
        self.ins_out = outconv(64, 32)  # instance classes=32 (it should be enough large to represent all instances)
        self.ins_cls_cnn = InstanceCounter(64, use_coordinates=True,
                                           usegpu=True)

    def forward(self, x):
        x1 = self.inc(x)  # (N, 64, h, w)

        x2 = self.down1(x1)  # (N, 128, h/2, w/2)
        x3 = self.down2(x2)  # (N, 256, h/4, w/4)
        x4 = self.down3(x3)  # (N, 512, h/8, w/8)
        x5 = self.down4(x4)  # (N, 512, h/16, w/16)

        x = self.up1(x5, x4)  # (N, 256, h/8, w/8)
        x = self.up2(x, x3)  # (N, 128, h/4, w/4)
        x = self.up3(x, x2)  # (N, 64, h/2, w/2)
        x = self.up4(x, x1)  # (N, 64, h, w)

        # heads
        sem = self.sem_out(x)  # (N, 2, h, w)
        ins = self.ins_out(x)  # (N, 32, h, w) - 32 dims embedding space
        cnt = self.ins_cls_cnn(x)

        return sem, ins, cnt


class UNet_CBAM_Deeper(nn.Module):
    '''Only added 1 down-path & corresponding up-path'''
    def __init__(self):
        super(UNet_CBAM_Deeper, self).__init__()
        self.inc = inconv(3, 64)

        self.down1 = down(64, 128)
        self.down2 = down(128, 256)
        self.down3 = down(256, 512)
        self.down4 = down(512, 1024)
        self.down5 = down(1024, 1024)

        self.up1 = up(2048, 512)
        self.up2 = up(1024, 256)
        self.up3 = up(512, 128)
        self.up4 = up(256, 64)
        self.up5 = up(128, 64)

        # Segmentation heads
        self.sem_out = outconv(64, 2)  # semantic classes=2 (f.g / b.g)
        self.ins_out = outconv(64, 4)  # instance classes=32 (enoguh large to represent all instances)
        self.ins_cls_cnn = InstanceCounter(64, use_coordinates=True,
                                           usegpu=True)

    def forward(self, x):
        x1 = self.inc(x)  # (N, 64, h, w)

        x2 = self.down1(x1)  # (N, 128, h/2, w/2)
        x3 = self.down2(x2)  # (N, 256, h/4, w/4)
        x4 = self.down3(x3)  # (N, 512, h/8, w/8)
        x5 = self.down4(x4)  # (N, 1024, h/16, w/16)
        x6 = self.down5(x5)  # (N, 1024, h/32, w/32)

        x = self.up1(x6, x5)  # (N, 512, h/16, w/16)
        x = self.up2(x, x4)  # (N, 256, h/8, w/8)
        x = self.up3(x, x3)  # (N, 128, h/4, w/4)
        x = self.up4(x, x2)  # (N, 64, h/2, w/2)
        x = self.up5(x, x1)  # (N, 64, h, w)

        sem = self.sem_out(x)  # (N, 2, h, w)
        ins = self.ins_out(x)  # (N, 32, h, w)
        cnt = self.ins_cls_cnn(x)

        return sem, ins, cnt

