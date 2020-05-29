import os
import sys
import random
from glob import glob
from itertools import cycle

from albumentations import Compose, BboxParams
from torch.utils.data import IterableDataset
import torchvision.transforms as transforms
import cv2
import numpy as np
import torch
from trvo_utils.imutils import imreadRGB

from utils.datasets import pad_to_square, resize


def files(dir, pattern):
    return glob(os.path.join(dir, pattern))


def _composeTransforms(transforms):
    bbox_params = BboxParams('yolo', ['class_ids'], min_visibility=.7)
    return Compose(transforms or [], bbox_params)


class MultiDirDataset(IterableDataset):
    def __init__(self, dataDirs, img_size, transforms=[], multiscale=True, normalized_labels=True):
        self.img_files = []
        self.label_files = []

        for dataDir in dataDirs:
            for labelFile in files(dataDir, '*.txt'):
                imgFound = False
                nameWithoutExt = os.path.splitext(labelFile)[0]
                if os.path.isfile(nameWithoutExt + '.jpg'):
                    self.img_files.append(nameWithoutExt + '.jpg')
                    imgFound = True
                elif os.path.isfile(nameWithoutExt + '.jpeg'):
                    self.img_files.append(nameWithoutExt + '.jpeg')
                    imgFound = True
                elif os.path.isfile(nameWithoutExt + '.png'):
                    self.img_files.append(nameWithoutExt + '.png')
                    imgFound = True
                if imgFound:
                    self.label_files.append(labelFile)

        assert len(self.label_files)
        self.img_size = img_size
        self.max_objects = 100
        self.transforms = _composeTransforms(transforms)
        self.multiscale = multiscale
        self.normalized_labels = normalized_labels
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __iter__(self):
        for i in cycle(range(len(self.img_files))):
            yield self.getitem(i)

    def getitem(self, index):
        img_path = self.img_files[index]

        # Extract image as PyTorch tensor
        img = imreadRGB(img_path)
        boxes = np.loadtxt(self.label_files[index]).reshape(-1, 5)
        r = self.transforms(image=img, bboxes=boxes[:, 1:], class_ids=boxes[:, 0])
        img = r['image']
        bb = r['bboxes']
        class_ids = np.array(r['class_ids'])
        boxes = np.hstack((class_ids.reshape(-1, 1), bb))

        img = transforms.ToTensor()(img)

        # Handle images with less than three channels
        if len(img.shape) != 3:
            img = img.unsqueeze(0)
            img = img.expand((3, img.shape[1:]))

        _, h, w = img.shape
        h_factor, w_factor = (h, w) if self.normalized_labels else (1, 1)
        # Pad to square resolution
        img, pad = pad_to_square(img, 0)
        _, padded_h, padded_w = img.shape

        # ---------
        #  Label
        # ---------

        boxes = torch.from_numpy(boxes)
        # Extract coordinates for unpadded + unscaled image
        x1 = w_factor * (boxes[:, 1] - boxes[:, 3] / 2)
        y1 = h_factor * (boxes[:, 2] - boxes[:, 4] / 2)
        x2 = w_factor * (boxes[:, 1] + boxes[:, 3] / 2)
        y2 = h_factor * (boxes[:, 2] + boxes[:, 4] / 2)
        # Adjust for added padding
        x1 += pad[0]
        y1 += pad[2]
        x2 += pad[1]
        y2 += pad[3]
        # Returns (x, y, w, h)
        boxes[:, 1] = ((x1 + x2) / 2) / padded_w
        boxes[:, 2] = ((y1 + y2) / 2) / padded_h
        boxes[:, 3] *= w_factor / padded_w
        boxes[:, 4] *= h_factor / padded_h

        targets = torch.zeros((len(boxes), 6))
        targets[:, 1:] = boxes

        return img_path, img, targets

    def collate_fn(self, batch):
        paths, imgs, targets = list(zip(*batch))
        # Remove empty placeholder targets
        targets = [boxes for boxes in targets if boxes is not None]
        # Add sample index to targets
        for i, boxes in enumerate(targets):
            boxes[:, 0] = i
        targets = torch.cat(targets, 0)
        # Selects new image size every tenth batch
        if self.multiscale and self.batch_count % 10 == 0:
            self.img_size = random.choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return paths, imgs, targets
