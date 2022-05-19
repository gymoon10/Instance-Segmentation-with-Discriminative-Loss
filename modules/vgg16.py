import torch.nn as nn
import torchvision.models as models
from modules.coord_conv import CoordConvNet

import torch
from torch import nn
from torch.autograd import Variable


class VGG16(nn.Module):

    r"""A module that augments VGG16 as defined in 'Very Deep Convolutional
    Networks for Large-Scale Image Recognition'
    (https://arxiv.org/pdf/1409.1556.pdf).

    1. It can return first `n_layers` of the VGG16.
    2. It can add coordinates to feature maps prior to each convolution.
    3. It can return all outputs (including intermediate outputs) of the
        VGG16.

    Args:
        n_layers (int): Use first `n_layers` layers of the VGG16
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds `x`, `y` and radius
            (`r`) coordinates to feature maps prior to each convolution.
            Weights to process these coordinates are initialized as zero.
            Default: `False`
        return_intermediate_outputs (bool, optional): If `True`, return
            outputs of the each layer in the VGG16 as a list otherwise
            return output of the last layer of first `n_layers` layers
            of the VGG16. Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: Output of the last layer of the selected subpart of VGG16
            or the list that contains outputs of the each layer depending on
            `return_intermediate_outputs`

    Examples:
        >>> vgg16 = VGG16(16, True, True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = vgg16(input)

        >>> vgg16 = VGG16(16, True, True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = vgg16(input)
    """

    def __init__(self, n_layers, pretrained=True, use_coordinates=False,
                 return_intermediate_outputs=False, usegpu=True):
        super(VGG16, self).__init__()

        self.use_coordinates = use_coordinates
        self.return_intermediate_outputs = return_intermediate_outputs

        self.cnn = models.__dict__['vgg16'](pretrained=pretrained)
        self.cnn = nn.Sequential(*list(self.cnn.children())[0])  # only extract feature extractors
        self.cnn = nn.Sequential(*list(self.cnn.children())[: n_layers])

        if self.use_coordinates:
            self.cnn = CoordConvNet(self.cnn, True, usegpu)  # cnn_model, with_r, usegpu

    def __get_outputs(self, x):
        '''save outputs of each conv layer'''
        if self.use_coordinates:
            return self.cnn(x)

        outputs = []
        for i, layer in enumerate(self.cnn.children()):
            x = layer(x)
            outputs.append(x)

        return outputs

    def forward(self, x):
        outputs = self.__get_outputs(x)

        if self.return_intermediate_outputs:
            return outputs  # all outputs of each conv layers

        return outputs[-1]  # only return final outputs of VGG16


class SkipVGG16(nn.Module):

    r"""A module that returns output of 7th convolutional layer of the
    VGG16 along with outputs of the 2nd and 4th convolutional layers.

    Args:
        pretrained (bool, optional): If `True`, initializes weights of the
            VGG16 using weights trained on ImageNet. Default: `True`
        use_coordinates (bool, optional): If `True`, adds `x`, `y` and radius
            (`r`) coordinates to feature maps prior to each convolution.
            Weights to process these coordinates are initialized as zero.
            Default: `False`
        usegpu (bool, optional): If `True`, runs operations on GPU
            Default: `True`

    Shape:
        - Input: `(N, C_{in}, H_{in}, W_{in})`
        - Output: List of outputs of the 2nd, 4th and 7th convolutional
            layers of the VGG16, respectively.

        - It seems that Output returns outputs of 2nd, 7th and last conv layers

    Examples:
        >>> vgg16 = SkipVGG16(True, True, False)
        >>> input = torch.randn(8, 3, 64, 64)
        >>> output = vgg16(input)

        >>> vgg16 = SkipVGG16(True, True, True).cuda()
        >>> input = torch.randn(8, 3, 64, 64).cuda()
        >>> output = vgg16(input)
    """

    def __init__(self, pretrained=True, use_coordinates=False,
                 usegpu=True):
        super(SkipVGG16, self).__init__()

        self.use_coordinates = use_coordinates

        self.outputs = [3, 8]
        self.n_filters = [64, 128]

        self.model = VGG16(n_layers=16, pretrained=True,  # use 0~15 layers of VGG16
                           use_coordinates=self.use_coordinates,
                           return_intermediate_outputs=True,  # return outputs of each conv layers
                           usegpu=usegpu)

    def forward(self, x):

        if self.use_coordinates:
            outs = self.model(x)
            out = [o for i, o in enumerate(outs) if i in self.outputs]
            out.append(outs[-1])

        else:
            out = []
            for i, layer in enumerate(list(self.model.children())[0]):  # 0 ~ 15 layers
                x = layer(x)
                if i in self.outputs:
                    out.append(x)
            out.append(x)  # outputs of 2nd, 7th, last conv layers

        return out

if __name__ == '__main__':
    from torch.autograd import Variable
    import torch
    import time

    def test(use_coordinates, usegpu, skip):

        batch_size, image_size = 4, 128

        if skip:
            vgg16 = SkipVGG16(False, use_coordinates, usegpu)
        else:
            vgg16 = VGG16(16, False, use_coordinates, False, usegpu)

        input = Variable(torch.rand(batch_size, 3, image_size,
                         image_size))

        if usegpu:
            vgg16 = vgg16.cuda()
            input = input.cuda()

        print('model :', vgg16)

        output = vgg16(input)

        if isinstance(output, list):
            print('N outputs : ', len(output))
            for o in output:
                print('** Output shape : ', o.size())
        else:
            print('\n** Output Shape : ', output.size())

    print('\n### COORDS + GPU + SKIP')
    test(True, True, True)
    print('\n### COORDS + GPU')
    test(True, True, False)
    print('\n### COORDS + CPU + SKIP')
    test(True, False, True)
    print('\n### COORDS + CPU')
    test(True, False, False)
    print('\n### GPU + SKIP')
    test(False, True, True)
    print('\n### GPU')
    test(False, True, False)
    print('\n### CPU + SKIP')
    test(False, False, True)
    print('\n### CPU')
    test(False, False, False)
