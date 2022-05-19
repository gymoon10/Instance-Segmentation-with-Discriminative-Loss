import torch
from torch.utils.data import Dataset
import random

import os
import glob

from PIL import Image
import sys
import numpy as np


class SegDataset(Dataset):
    """Dataset Reader
       instance annotations: (N, 20=MAX_N_OBJECTS, h, w)
         - each channel-dim of instance annotations corresponds to each leaf instances
         - if actual # of instances < MAX_N_OBJECTS -> remainders are filled with zero"""

    def __init__(self, image_list, semantic_annotation_list, instance_annotation_list):
        self.images = image_list
        self.semantics = semantic_annotation_list
        self.instances = instance_annotation_list

    def __len__(self):
        return len(self.images)

    def __load_data(self, index):
        img = Image.open(self.images[index])  # RGBA PIL

        semantic_annotation = np.load(self.semantics[index])  # dtype=uint8 (h, w)
        instance_annotation = np.load(self.instances[index])  # dtype=uint8 (h, w, GT_n_objects)

        n_objects = instance_annotation.shape[-1]  # # of leaf object instances
        height = instance_annotation.shape[0]
        width = instance_annotation.shape[1]

        return img, semantic_annotation, instance_annotation, n_objects

    def __getitem__(self, index):
        assert index <= len(self), 'index range error'

        image, semantic_annotation, instance_annotation, n_objects = self.__load_data(index)

        return image, semantic_annotation, instance_annotation, n_objects


from PIL import Image
import torchvision.transforms as transforms

import data.preprocess2 as preprocess
from data.utils import ImageUtilities as IU


class AlignCollate(object):

    def __init__(self, mode, n_classes, max_n_objects, mean, std, image_height,
                 image_width, random_hor_flipping=True,
                 random_ver_flipping=True, random_transposing=True,
                 random_90x_rotation=True, random_rotation=True,
                 random_color_jittering=False, random_grayscaling=False,
                 random_channel_swapping=False, random_gamma=False,
                 random_resolution=True):

        self._mode = mode
        self.n_classes = n_classes
        self.max_n_objects = max_n_objects

        assert self._mode in ['training', 'test']

        self.mean = mean
        self.std = std
        self.image_height = image_height
        self.image_width = image_width

        self.random_horizontal_flipping = random_hor_flipping
        self.random_vertical_flipping = random_ver_flipping
        self.random_transposing = random_transposing
        self.random_90x_rotation = random_90x_rotation
        self.random_rotation = random_rotation
        self.random_color_jittering = random_color_jittering
        self.random_grayscaling = random_grayscaling
        self.random_channel_swapping = random_channel_swapping
        self.random_gamma = random_gamma
        self.random_resolution = random_resolution

        if self._mode == 'training':
            if self.random_horizontal_flipping:
                self.horizontal_flipper = IU.image_random_horizontal_flipper()
            if self.random_vertical_flipping:
                self.vertical_flipper = IU.image_random_vertical_flipper()
            if self.random_transposing:
                self.transposer = IU.image_random_transposer()
            if self.random_rotation:
                self.image_rotator = IU.image_random_rotator(random_bg=True)
                self.annotation_rotator = IU.image_random_rotator(Image.NEAREST,
                                                                  random_bg=False)
            if self.random_90x_rotation:
                self.image_rotator_90x = IU.image_random_90x_rotator()
                self.annotation_rotator_90x = IU.image_random_90x_rotator(Image.NEAREST)
            if self.random_color_jittering:
                self.color_jitter = IU.image_random_color_jitter(
                    brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)
            if self.random_grayscaling:
                self.grayscaler = IU.image_random_grayscaler(p=0.3)
            if self.random_channel_swapping:
                self.channel_swapper = IU.image_random_channel_swapper(p=0.5)
            if self.random_gamma:
                self.gamma_adjuster = IU.image_random_gamma([0.7, 1.3], gain=1)
            if self.random_resolution:
                self.resolution_degrader = IU.image_random_resolution([0.7, 1.3])

            self.img_resizer = IU.image_resizer(self.image_height,
                                                self.image_width)
            self.ann_resizer = IU.image_resizer(self.image_height,
                                                self.image_width,
                                                interpolation=Image.NEAREST)
        else:
            self.img_resizer = IU.image_resizer(self.image_height,
                                                self.image_width)
            self.ann_resizer = IU.image_resizer(self.image_height,
                                                self.image_width,
                                                interpolation=Image.NEAREST)

        self.image_normalizer = IU.image_normalizer(self.mean, self.std)

    def __preprocess(self, image, semantic_annotation, instance_annotation):
        # Augmentation
        if self._mode == 'training':
            instance_annotation = list(instance_annotation.transpose(2, 0, 1))
            n_objects = len(instance_annotation)

            if self.random_resolution:
                image = self.resolution_degrader(image)

            if self.random_horizontal_flipping:
                is_flip = random.random() < 0.5
                image = self.horizontal_flipper(image, is_flip)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.horizontal_flipper(_ann, is_flip)
                    instance_annotation[i] = _ann

                semantic_annotation = self.horizontal_flipper(
                    semantic_annotation, is_flip)

            if self.random_vertical_flipping:
                is_flip = random.random() < 0.5
                image = self.vertical_flipper(image, is_flip)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.vertical_flipper(_ann, is_flip)
                    instance_annotation[i] = _ann

                semantic_annotation = self.vertical_flipper(
                    semantic_annotation, is_flip)

            if self.random_transposing:
                is_trans = random.random() < 0.5
                image = self.transposer(image, is_trans)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.transposer(_ann, is_trans)
                    instance_annotation[i] = _ann

                semantic_annotation = self.transposer(
                    semantic_annotation, is_trans)

            if self.random_90x_rotation:
                rot_angle = np.random.choice([0, 90, 180, 270])
                rot_expand = True
                image = self.image_rotator_90x(image, rot_angle, rot_expand)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.annotation_rotator_90x(_ann, rot_angle, rot_expand)
                    instance_annotation[i] = _ann

                semantic_annotation = self.annotation_rotator_90x(semantic_annotation,
                                                                  rot_angle, rot_expand)

            if self.random_rotation:
                rot_angle = int(np.random.rand() * 10)
                if np.random.rand() >= 0.5:
                    rot_angle = -1 * rot_angle
                # rot_expand = np.random.rand() < 0.5
                rot_expand = True
                image = self.image_rotator(image, rot_angle, rot_expand)

                for i in range(n_objects):
                    _ann = instance_annotation[i].copy()
                    _ann = self.annotation_rotator(_ann, rot_angle, rot_expand)
                    instance_annotation[i] = _ann

                semantic_annotation = self.annotation_rotator(semantic_annotation,
                                                              rot_angle, rot_expand)

            if self.random_color_jittering:
                image = self.color_jitter(image)

            if self.random_gamma:
                image = self.gamma_adjuster(image)

            if self.random_channel_swapping:
                image = self.channel_swapper(image)

            if self.random_grayscaling:
                image = self.grayscaler(image)

            instance_annotation = np.array(
                instance_annotation).transpose(1, 2, 0)

        # Resize Images
        image = self.img_resizer(image)

        # Resize Instance Annotations
        ann_height, ann_width, n_objects = instance_annotation.shape
        instance_annotation_resized = []

        height_ratio = 1.0 * self.image_height / ann_height
        width_ratio = 1.0 * self.image_width / ann_width

        for i in range(n_objects):
            instance_ann_img = Image.fromarray(instance_annotation[:, :, i])
            instance_ann_img = self.ann_resizer(instance_ann_img)
            instance_ann_img = np.array(instance_ann_img)

            instance_annotation_resized.append(instance_ann_img)

        # Fill Instance Annotations with zeros
        for i in range(self.max_n_objects - n_objects):
            zero = np.zeros((ann_height, ann_width),
                            dtype=np.uint8)
            zero = Image.fromarray(zero)
            zero = self.ann_resizer(zero)
            zero = np.array(zero)
            instance_annotation_resized.append(zero.copy())

        instance_annotation_resized = np.stack(
            instance_annotation_resized, axis=0)
        instance_annotation_resized = instance_annotation_resized.transpose(
            1, 2, 0)

        # Resize Semantic Annotations
        semantic_annotation = self.ann_resizer(
            Image.fromarray(semantic_annotation))
        semantic_annotation = np.array(semantic_annotation)

        # Image Normalization (only converts PIL to Tensor, check data.utils)
        image = self.image_normalizer(image.convert('RGB'))

        return (image, semantic_annotation, instance_annotation_resized)

    def __call__(self, batch):
        images, semantic_annotations, instance_annotations, \
        n_objects = zip(*batch)

        images = list(images)
        semantic_annotations = list(semantic_annotations)
        instance_annotations = list(instance_annotations)

        # max_n_objects = np.max(n_objects)

        bs = len(images)
        for i in range(bs):
            image, semantic_annotation, instance_annotation = \
                self.__preprocess(images[i],
                                  semantic_annotations[i],
                                  instance_annotations[i])

            images[i] = image
            semantic_annotations[i] = semantic_annotation
            instance_annotations[i] = instance_annotation

        images = torch.stack(images)

        instance_annotations = np.array(
            instance_annotations,
            dtype='int')  # bs, h, w, n_ins

        semantic_annotations = np.array(
            semantic_annotations, dtype='int')  # bs, h, w
        semantic_annotations_one_hot = np.eye(self.n_classes, dtype='int')
        semantic_annotations_one_hot = \
            semantic_annotations_one_hot[semantic_annotations.flatten()].reshape(
                semantic_annotations.shape[0], semantic_annotations.shape[1],
                semantic_annotations.shape[2], self.n_classes)

        instance_annotations = torch.LongTensor(instance_annotations)
        instance_annotations = instance_annotations.permute(0, 3, 1, 2)

        semantic_annotations_one_hot = torch.LongTensor(
            semantic_annotations_one_hot)
        semantic_annotations_one_hot = semantic_annotations_one_hot.permute(
            0, 3, 1, 2)

        n_objects = torch.IntTensor(n_objects)

        return (images, semantic_annotations_one_hot, instance_annotations,
                n_objects)

