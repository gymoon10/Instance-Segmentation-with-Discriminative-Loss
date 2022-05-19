import torch
from torch.utils.data import Dataset
import random

import os
import glob

from PIL import Image
import sys
import numpy as np
from dataset2 import SegDataset, AlignCollate

# To test dataset2.py

DATA_DIR = 'E:/CVPPP2017_LSC_training/images'

image_list = glob.glob(os.path.join(DATA_DIR, 'A1', '*_rgb.png'))
semantic_annotation_list = glob.glob(os.path.join(DATA_DIR, 'processed', 'CVPPP', 'semantic-annotations', '*.npy'))
instance_annotation_list = glob.glob(os.path.join(DATA_DIR, 'processed', 'CVPPP', 'instance-annotations', '*.npy'))

ds = SegDataset(image_list, semantic_annotation_list, instance_annotation_list)

image, semantic_annotation, instance_annotation, n_objects = ds[5]

print(image.size)
print(semantic_annotation.shape)
print(instance_annotation.shape)
print(n_objects)
print(np.unique(semantic_annotation))
print(np.unique(instance_annotation))

ac = AlignCollate('training', 2, 20, [0.0, 0.0, 0.0],
                   [1.0, 1.0, 1.0], 256, 256)

loader = torch.utils.data.DataLoader(ds, batch_size=1,
                                     shuffle=False,
                                     num_workers=0,
                                     pin_memory=False,
                                     collate_fn=ac)

loader = iter(loader)

images, semantic_annotations, instance_annotations, n_objects = loader.next()

print('Input image: ', images.size())
print('Input image-data type: ', type(images))
print('GT-semantic mask: ', semantic_annotations.size())
print('GT-instance mask :', instance_annotations.size())
print('GT-# of instances: ', n_objects)