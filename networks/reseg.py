import torch
import torch.nn as nn
from modules.vgg16 import SkipVGG16
from modules.renet import ReNet
from modules.instance_counter import InstanceCounter


class ReSeg(nn.Module):

    r"""ReSeg Module (with modifications) as defined in 'ReSeg: A Recurrent
    Neural Network-based Model for Semantic Segmentation'
    (https://arxiv.org/pdf/1511.07053.pdf).

    * VGG16 with skip Connections as base network
    * Two ReNet layers
    * Two transposed convolutional layers for upsampling
    * Three heads for semantic segmentation, instance segmentation and
        instance counting.

    Args:
        n_classes (int): Number of semantic classes
        use_instance_seg (bool, optional): If `False`, does not perform
            instance segmentation. Default: `True`
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds coordinate
            information to input image and hidden state. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output:
            - Semantic Seg: `(N, N_{class}, H_{in}, W_{in})`
            - Instance Seg: `(N, 32, H_{in}, W_{in}) - 개별 Instance mask를 예측하기 위한 32 채널 (Max_N_OBJECTS보다 충분히 크게)`
            - Instance Cnt: `(N, 1)`

    Examples:
        >>> reseg = ReSeg(3, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> outputs = reseg(input)

        >>> reseg = ReSeg(3, True, True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> outputs = reseg(input)
    """

    def __init__(self, n_classes, use_instance_seg=True, pretrained=True,
                 use_coordinates=False, usegpu=True):
        super(ReSeg, self).__init__()

        self.n_classes = n_classes  # 2 (for semantic segmentation)
        self.use_instance_seg = use_instance_seg

        # Encoder
        # BaseCNN
        self.cnn = SkipVGG16(pretrained=pretrained,
                             use_coordinates=use_coordinates,
                             usegpu=usegpu)

        # ReNets
        self.renet1 = ReNet(256, 100, use_coordinates=use_coordinates,
                            usegpu=usegpu)  # n_input=256, n_units=100
        self.renet2 = ReNet(100 * 2, 100, use_coordinates=use_coordinates,
                            usegpu=usegpu)

        # Decoder
        self.upsampling1 = nn.ConvTranspose2d(100 * 2, 100,
                                              kernel_size=(2, 2),
                                              stride=(2, 2))
        self.relu1 = nn.ReLU()
        self.upsampling2 = nn.ConvTranspose2d(100 + self.cnn.n_filters[1],
                                              100, kernel_size=(2, 2),
                                              stride=(2, 2))
        self.relu2 = nn.ReLU()

        # Semantic Segmentation
        self.sem_seg_output = nn.Conv2d(100 + self.cnn.n_filters[0],
                                        self.n_classes, kernel_size=(1, 1),
                                        stride=(1, 1))

        # Instance Segmentation
        if self.use_instance_seg:
            self.ins_seg_output = nn.Conv2d(100 + self.cnn.n_filters[0],
                                            32, kernel_size=(1, 1),
                                            stride=(1, 1))

        # Instance Counting (CNN Network)
        self.ins_cls_cnn = InstanceCounter(100 * 2, use_coordinates,
                                           usegpu=usegpu)

    def forward(self, x):

        # Encoder
        # BaseCNN (SkipVgg16)
        first_skip, second_skip, x_enc = self.cnn(x)  # outputs of 2nd, 7th, last conv layers
        # fist_skip: (N, 64, h, w)
        # second_skip: (N, 128, h/2, w/2)
        # x_enc: (N, 256, h/4, w/4)

        # ReNets
        x_enc = self.renet1(x_enc)
        x_enc = self.renet2(x_enc)  # (N, 200, 256, 256)

        # Decoder
        x_dec = self.relu1(self.upsampling1(x_enc))
        x_dec = torch.cat((x_dec, second_skip), dim=1)
        x_dec = self.relu2(self.upsampling2(x_dec))
        x_dec = torch.cat((x_dec, first_skip), dim=1)  # (N, 164, 256, 256)

        # Semantic Segmentation
        sem_seg_out = self.sem_seg_output(x_dec)  # (N, 2, 256, 256)

        # Instance Segmentation
        if self.use_instance_seg:
            ins_seg_out = self.ins_seg_output(x_dec)  # (N, 32, 256, 256)
        else:
            ins_seg_out = None

        # Instance Counting
        ins_cls_out = self.ins_cls_cnn(x_enc)

        return sem_seg_out, ins_seg_out, ins_cls_out


#reseg = ReSeg(3, True, True, True, False)
#input = torch.randn(8, 3, 256, 256)
#seg, ins, cnt = reseg(input)

#print(seg.shape)
#print(ins.shape)
#print(x_enc.shape)