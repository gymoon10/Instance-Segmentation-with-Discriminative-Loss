import matplotlib
matplotlib.use('Agg')

import matplotlib.pylab as plt
import os
import sys
import argparse
import numpy as np
from PIL import Image


import argparse
import random
import os
import glob
import getpass
import datetime
import shutil
import numpy as np
import torch
from argparse import Namespace
from os import listdir
from os.path import join

from model import Model
from prediction import Prediction
from settings.training_settings import TrainingSettings as CVPPPTrainingSettings


opt= Namespace()
opt.usegpu = True
opt.debug = False
opt.nworkers = 0
opt.dataset = 'CVPPP2017'
opt.DATA_DIR = 'E:/CVPPP2017_LSC_training/images'
opt.model = 'E:/discriminative/models/cbam_4/best.pth'  # model weights after Train

model_path = opt.model
_output_path = 'E:/discriminative/models/cbam_4/Outputs'  # inference results are saved

basepath = 'E:/CVPPP2017_LSC_training/images/val2'  # path of validation image folder
image_names = sorted([join(basepath, f) for f in listdir(basepath) if f.endswith('_rgb.png')])

# can change settings at settings
ts = CVPPPTrainingSettings()

model = Model('CVPPP', ts.MODEL_NAME, ts.N_CLASSES, ts.MAX_N_OBJECTS,
              use_instance_segmentation=ts.USE_INSTANCE_SEGMENTATION,
              use_coords=ts.USE_COORDINATES, load_model_path=opt.model,
              usegpu=opt.usegpu)

prediction = Prediction(ts.IMAGE_HEIGHT, ts.IMAGE_WIDTH,
                        ts.MEAN, ts.STD, ts.USE_COORDINATES, model,
                        0)

for image_name in image_names:
    image, fg_seg_pred, ins_seg_pred, n_objects_pred = \
        prediction.predict(image_name)

    #_output_path = os.path.join(output_path, image_name)
    print('output path :', _output_path)

    try:
        os.makedirs(_output_path)
    except BaseException:
        pass

    fg_seg_pred = fg_seg_pred * 255

    _n_clusters = len(np.unique(ins_seg_pred.flatten())) - 1  # discard bg
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, _n_clusters)]
    ins_seg_pred_color = np.zeros(
        (ins_seg_pred.shape[0], ins_seg_pred.shape[1], 3), dtype=np.uint8)
    for i in range(_n_clusters):
        ins_seg_pred_color[ins_seg_pred == (
            i + 1)] = (np.array(colors[i][:3]) * 255).astype('int')

    image_pil = Image.fromarray(image)
    fg_seg_pred_pil = Image.fromarray(fg_seg_pred)
    ins_seg_pred_pil = Image.fromarray(ins_seg_pred)
    ins_seg_pred_color_pil = Image.fromarray(ins_seg_pred_color)

    image_pil.save(os.path.join(_output_path, image_name[-16:] + '.png'))
    fg_seg_pred_pil.save(os.path.join(_output_path, image_name[-16:] + '-fg_mask.png'))
    ins_seg_pred_pil.save(os.path.join(_output_path, image_name[-16:] + '-ins_mask.png'))
    ins_seg_pred_color_pil.save(os.path.join(
        _output_path, image_name[-16:] + '-ins_mask_color.png'))
    np.save(
        os.path.join(
            _output_path,
            image_name[-16:] +
            '-n_objects.npy'),
        n_objects_pred)