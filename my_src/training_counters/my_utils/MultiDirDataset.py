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
from trvo_utils import toInt
from trvo_utils.annotation import PascalVocXmlParser, BBox
from trvo_utils.imutils import imreadRGB, imSize, imshowWait
from trvo_utils.iter_utils import unzip

from utils.datasets import pad_to_square, resize


def _composeTransforms(transforms):
    bbox_params = BboxParams('pascal_voc', ['class_ids'], min_visibility=.5)
    return Compose(transforms or [], bbox_params)


class _Loader:
    imgExts = ['.jpg', '.jpeg', '.png']

    def __init__(self, dataDirs, labelNames):
        assert len(labelNames)
        self.labelNames = labelNames
        self.dataDirs = dataDirs

    @staticmethod
    def _files(dir, pattern):
        return glob(os.path.join(dir, pattern))

    def _annotatedImages(self):
        for dataDir in self.dataDirs:
            for labelFile in self._files(dataDir, '*.xml'):
                nameWithoutExt = os.path.splitext(labelFile)[0]
                imgFile = next(
                    (nameWithoutExt + imExt for imExt in self.imgExts if os.path.isfile(nameWithoutExt + imExt)),
                    "")
                if imgFile:
                    yield imgFile, labelFile

    def _parseAnnotations(self, labelFile):
        p = PascalVocXmlParser(labelFile)
        class_ids = [self.labelNames.index(l) for l in p.labels()]
        desiredAnnotations = ((b, l) for b, l in zip(p.boxes(), class_ids) if l != -1)
        boxes, class_ids = unzip(desiredAnnotations, [], [])
        return boxes, class_ids

    def loadItems(self):
        items = []
        for imgFile, labelFile in self._annotatedImages():
            img = imreadRGB(imgFile)
            if len(img.shape) == 2:
                img = np.dstack([img] * 3)
            boxes, class_ids = self._parseAnnotations(labelFile)
            item = imgFile, img, boxes, class_ids
            items.append(item)
        assert len(items)
        return items


_emptyTargets = np.empty([0, 5], np.float32)


class MultiDirDataset(IterableDataset):
    def __init__(self, dataDirs, img_size, label_names, transforms=None, multiscale=True):
        self.items = _Loader(dataDirs, label_names).loadItems()
        self.img_size = img_size

        if transforms is None:
            self.transforms = None
            self.infinite = False
        else:
            self.transforms = _composeTransforms(transforms)
            self.infinite = True

        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __iter__(self):
        if self.infinite:
            for i in cycle(range(len(self.items))):
                yield self.getitem(i)
        else:
            for i in range(len(self.items)):
                yield self.getitem(i)

    def _transformItem(self, index):
        _, img, boxes, class_ids = self.items[index]

        # imgDebug = img.copy()
        # for b in boxes:
        #     x1, y1, x2, y2  =toInt(*b)
        #     cv2.rectangle(imgDebug, (x1, y1), (x2, y2), (255, 255, 255), 2)
        # imshowWait(cv2.resize(imgDebug, None, None, .5, .5))

        if self.transforms:
            r = self.transforms(image=img, bboxes=boxes, class_ids=class_ids)
            img = r['image']
            boxes = r['bboxes']
            class_ids = r['class_ids']
            assert len(boxes) == len(class_ids)

        targets = _emptyTargets
        if len(boxes):
            boxes = BBox.voc2yolo_boxes(boxes, imSize(img))
            targets = np.insert(boxes, 0, class_ids, 1)

        return img, targets

    def getitem(self, index):
        img, boxes = self._transformItem(index)
        img = transforms.ToTensor()(img)

        _, h, w = img.shape
        h_factor, w_factor = h, w
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

        return img, targets

    def collate_fn(self, batch):
        imgs, targets = list(zip(*batch))
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
        return imgs, targets
