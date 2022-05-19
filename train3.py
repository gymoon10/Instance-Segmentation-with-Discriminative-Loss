import argparse
import random
import os
import glob
import getpass
import datetime
import shutil
import numpy as np
import torch
from data.dataset2 import SegDataset, AlignCollate
from model import Model
from argparse import Namespace
from settings.training_settings import TrainingSettings as CVPPPTrainingSettings

opt = Namespace()
opt.usegpu = True
opt.nepochs = 500
opt.batchsize = 4
opt.debug = False
opt.nworkers = 0
opt.dataset = 'CVPPP2017'
opt.DATA_DIR = 'E:/CVPPP2017_LSC_training/images'
opt.model = ''

# can change settings at settings
ts = CVPPPTrainingSettings()

# Train Data
train_image_list = glob.glob(os.path.join(opt.DATA_DIR, 'train2', '*_rgb.png'))
train_semantic_annotation_list = glob.glob(os.path.join(opt.DATA_DIR, 'processed', 'CVPPP', 'train-semantic-annotations2', '*.npy'))
train_instance_annotation_list = glob.glob(os.path.join(opt.DATA_DIR, 'processed', 'CVPPP', 'train-instance-annotations2', '*.npy'))

# Val Data
val_image_list = glob.glob(os.path.join(opt.DATA_DIR, 'val2', '*_rgb.png'))
val_semantic_annotation_list = glob.glob(os.path.join(opt.DATA_DIR, 'processed', 'CVPPP', 'val-semantic-annotations2', '*.npy'))
val_instance_annotation_list = glob.glob(os.path.join(opt.DATA_DIR, 'processed', 'CVPPP', 'val-instance-annotations2', '*.npy'))


def generate_run_id():

    username = getpass.getuser()

    now = datetime.datetime.now()
    date = map(str, [now.year, now.month, now.day])
    coarse_time = map(str, [now.hour, now.minute])
    fine_time = map(str, [now.second, now.microsecond])

    run_id = '_'.join(['-'.join(date), '-'.join(coarse_time),
                       username, '-'.join(fine_time)])
    return run_id

RUN_ID = generate_run_id()
print(RUN_ID)

model_save_path = f'E:/discriminative/models/{RUN_ID}'
os.makedirs(model_save_path)

if torch.cuda.is_available() and not opt.usegpu:
    print('WARNING: You have a CUDA device, so you should probably \
        run with --usegpu')

# Load Seeds
random.seed(ts.SEED)
np.random.seed(ts.SEED)
torch.manual_seed(ts.SEED)

# Define Data Loaders
pin_memory = False
if opt.usegpu:
    pin_memory = True

train_dataset = SegDataset(train_image_list, train_semantic_annotation_list, train_instance_annotation_list)
assert train_dataset

ac = AlignCollate('training', 2, 20, [0.521697844321, 0.389775426267, 0.206216114391],
                   [0.212398291819, 0.151755427041, 0.113022107204], 256, 256)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batchsize,
                                           shuffle=True,
                                           num_workers=0,
                                           pin_memory=False,
                                           collate_fn=ac)

#train_loader = iter(train_loader)

#images, semantic_annotations, instance_annotations, n_objects = train_loader.next()

#print('TRAIN')
#print('image: ', images.size())
#print('semantic: ', semantic_annotations.size())
#print('instance :', instance_annotations.size())
#print(n_objects.size())
#print('n_objects', n_objects)

test_dataset = SegDataset(val_image_list, val_semantic_annotation_list, val_instance_annotation_list)
assert test_dataset

test_align_collate = AlignCollate(
    'test',
    ts.N_CLASSES,
    ts.MAX_N_OBJECTS,
    ts.MEAN,
    ts.STD,
    ts.IMAGE_HEIGHT,
    ts.IMAGE_WIDTH,
    random_hor_flipping=ts.HORIZONTAL_FLIPPING,
    random_ver_flipping=ts.VERTICAL_FLIPPING,
    random_transposing=ts.TRANSPOSING,
    random_90x_rotation=ts.ROTATION_90X,
    random_rotation=ts.ROTATION,
    random_color_jittering=ts.COLOR_JITTERING,
    random_grayscaling=ts.GRAYSCALING,
    random_channel_swapping=ts.CHANNEL_SWAPPING,
    random_gamma=ts.GAMMA_ADJUSTMENT,
    random_resolution=ts.RESOLUTION_DEGRADING)

test_loader = torch.utils.data.DataLoader(test_dataset,
                                          batch_size=opt.batchsize,
                                          shuffle=False,
                                          num_workers=opt.nworkers,
                                          pin_memory=pin_memory,
                                          collate_fn=test_align_collate)

#test_loader = iter(test_loader)

#images, semantic_annotations, instance_annotations, n_objects = test_loader.next()

#print('TEST')
#print('image: ', images.size())
#print('semantic: ', semantic_annotations.size())
#print('instance :', instance_annotations.size())
#print(n_objects.size())
#print('n_objects', n_objects)


# IMPORTANT! : 'python -m visdom.server' should be executed at terminal
model = Model('CVPPP', ts.MODEL_NAME, ts.N_CLASSES, ts.MAX_N_OBJECTS,
              use_instance_segmentation=ts.USE_INSTANCE_SEGMENTATION,
              use_coords=ts.USE_COORDINATES, load_model_path=opt.model,
              usegpu=opt.usegpu)

print('Model Loaded')


model.fit(ts.CRITERION, ts.DELTA_VAR, ts.DELTA_DIST, ts.NORM, ts.LEARNING_RATE,
          ts.WEIGHT_DECAY, ts.CLIP_GRAD_NORM, ts.LR_DROP_FACTOR,
          ts.LR_DROP_PATIENCE, ts.OPTIMIZE_BG, ts.OPTIMIZER, ts.TRAIN_CNN,
          opt.nepochs, ts.CLASS_WEIGHTS, train_loader, test_loader,
          model_save_path, opt.debug)