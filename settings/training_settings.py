import os
from settings.model_settings import ModelSettings


class TrainingSettings(ModelSettings):

    def __init__(self):
        super(TrainingSettings, self).__init__()

        # if train_cnn=True -> cnn parameters(VGG16 part) are trained
        # if train_cnn=False -> cnn parameters are fixed
        self.TRAIN_CNN = True

        self.OPTIMIZER = 'Adadelta'
        # optimizer - one of : 'RMSprop', 'Adam', 'Adadelta', 'SGD'
        self.LEARNING_RATE = 1.0
        self.LR_DROP_FACTOR = 0.1
        self.LR_DROP_PATIENCE = 20
        self.WEIGHT_DECAY = 0.001
        # weight decay - use 0 to disable it
        self.CLIP_GRAD_NORM = 10.0
        # max l2 norm of gradient of parameters - use 0 to disable it

        # Options below are for Validation data actually
        # All True for train data except COLOR_JITTERING~GAMMA_ADJUSTMENT
        # As I confirmed that using those 4 options and image normalize severly degrades image quality
        self.HORIZONTAL_FLIPPING = False
        self.VERTICAL_FLIPPING = False
        self.TRANSPOSING = False
        self.ROTATION_90X = False
        self.ROTATION = False
        self.COLOR_JITTERING = False  # not used for training data also
        self.GRAYSCALING = False  # not used for training data also
        self.CHANNEL_SWAPPING = False  # not used for training data also
        self.GAMMA_ADJUSTMENT = False  # not used for training data also
        self.RESOLUTION_DEGRADING = False

        self.CRITERION = 'Multi'
        # criterion - One of 'CE', 'Dice', 'Multi'
        self.OPTIMIZE_BG = False

        # self.RANDOM_CROPPING = False
        # CROP_SCALE and CROP_AR is used iff self.RANDOM_CROPPING is True
        # self.CROP_SCALE = (1.0, 1.0)
        # Choose it carefully - have a look at
        # lib/preprocess.py -> RandomResizedCrop
        # self.CROP_AR = (1.0, 1.0)
        # Choose it carefully - have a look
        # at lib/preprocess.py -> RandomResizedCrop

        self.SEED = 23
