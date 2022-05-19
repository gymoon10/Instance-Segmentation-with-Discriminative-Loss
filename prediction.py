import torch.nn as nn
import numpy as np
from PIL import Image
import cv2
from sklearn.cluster import KMeans
from data.utils import ImageUtilities
from modules.coord_conv import AddCoordinates
import torchvision.transforms as transforms


class Prediction(object):

    def __init__(self, resize_height, resize_width, mean,
                 std, use_coordinates, model, n_workers=0):

        self.normalizer = ImageUtilities.image_normalizer(mean, std)
        self.use_coordinates = use_coordinates

        self.resize_height = resize_height
        self.resize_width = resize_width
        self.model = model

        self.n_workers = n_workers

        self.img_resizer = ImageUtilities.image_resizer(
            self.resize_height, self.resize_width)

        if self.use_coordinates:
            self.coordinate_adder = AddCoordinates(with_r=True,
                                                   usegpu=False)

    def get_image(self, image_path):

        img = ImageUtilities.read_image(image_path)  # open image as PIL RGB
        image_width, image_height = img.size

        img = self.img_resizer(img)
        img = self.normalizer(img)  # PIL to Tensor actually

        return img, image_height, image_width

    def get_annotation(self, annotation_path):

        img = ImageUtilities.read_image(annotation_path)
        return img

    def upsample_prediction(self, prediction, image_height, image_width):

        return cv2.resize(prediction, (image_width, image_height),
                          interpolation=cv2.INTER_NEAREST)

    def cluster(self, sem_seg_prediction, ins_seg_prediction,
                n_objects_prediction):

        seg_height, seg_width = ins_seg_prediction.shape[1:]

        sem_seg_prediction = sem_seg_prediction.cpu().numpy()
        sem_seg_prediction = sem_seg_prediction.argmax(0).astype(np.uint8)

        embeddings = ins_seg_prediction.cpu()  # (32, 256, 256)

        if self.use_coordinates:
            embeddings = embeddings.unsqueeze(0)  # add batch dim (1, 32, 256, 256)
            # AddCoordinates(with_r=True, usegpu=False)
            embeddings = self.coordinate_adder(embeddings)  # (1, 32+3, 256, 256)
            embeddings = embeddings.squeeze(0)
            
        embeddings = embeddings.numpy()
        embeddings = embeddings.transpose(1, 2, 0)  # h, w, c

        n_objects_prediction = n_objects_prediction.cpu().numpy()[0]

        # embeddings[:, :, i][sem_seg_prediction != 0] + embeddings[:, :, i][sem_seg_prediction == 0] = h x w
        # from each dim of embedding, pixel values of RoI (semantic mask) are extracted
        embeddings = np.stack([embeddings[:, :, i][sem_seg_prediction != 0]
                               for i in range(embeddings.shape[2])], axis=1)

        labels = KMeans(n_clusters=n_objects_prediction,
                        n_init=35, max_iter=500).fit_predict(embeddings)

        instance_mask = np.zeros((seg_height, seg_width), dtype=np.uint8)

        fg_coords = np.where(sem_seg_prediction != 0)
        for si in range(len(fg_coords[0])):
            y_coord = fg_coords[0][si]
            x_coord = fg_coords[1][si]
            _label = labels[si] + 1
            instance_mask[y_coord, x_coord] = _label

        return sem_seg_prediction, instance_mask, n_objects_prediction

    def predict(self, image_path):

        image, image_height, image_width = self.get_image(image_path)  # (3, h, w)
        image = image.unsqueeze(0)  # add batch dim

        sem_seg_prediction, ins_seg_prediction, n_objects_prediction = \
            self.model.predict(image)

        sem_seg_prediction = sem_seg_prediction.squeeze(0)  # (2, 256, 256)
        ins_seg_prediction = ins_seg_prediction.squeeze(0)  # (32, 256, 256)
        n_objects_prediction = n_objects_prediction.squeeze(0)

        sem_seg_prediction, ins_seg_prediction, \
            n_objects_prediction = self.cluster(sem_seg_prediction,
                                                ins_seg_prediction,
                                                n_objects_prediction)

        sem_seg_prediction = self.upsample_prediction(
            sem_seg_prediction, image_height, image_width)
        ins_seg_prediction = self.upsample_prediction(
            ins_seg_prediction, image_height, image_width)

        raw_image_pil = ImageUtilities.read_image(image_path)
        raw_image = np.array(raw_image_pil)

        return raw_image, sem_seg_prediction, ins_seg_prediction, \
            n_objects_prediction
