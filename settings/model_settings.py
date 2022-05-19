import os
import numpy as np
from settings.data_settings import DataSettings


class ModelSettings(DataSettings):

    def __init__(self):
        super(ModelSettings, self).__init__()

        # self.MEAN = [0.485, 0.456, 0.406]
        # self.STD = [0.229, 0.224, 0.225]
        self.MEAN = [0.521697844321, 0.389775426267, 0.206216114391]
        self.STD = [0.212398291819, 0.151755427041, 0.113022107204]

        self.MODEL_NAME = 'UNet_CBAM_Deeper'  # 'ReSeg' / DeepLabV3 / 'StackedRecurrentHourglass' / SegNet

        self.USE_INSTANCE_SEGMENTATION = True  # for Instance Segmentation
        self.USE_COORDINATES = False  # use coord-conv or not

        self.IMAGE_HEIGHT = 256
        self.IMAGE_WIDTH = 256

        # hyper-parameters for discriminative loss
        self.DELTA_VAR = 0.5
        self.DELTA_DIST = 1.5
        self.NORM = 2
