import os
from glob import glob
from itertools import repeat
from random import choice, choices
import numpy as np

import torch
from albumentations import BboxParams, Compose
from torch.utils.data import IterableDataset
from trvo_utils.annotation import BBox
from trvo_utils.imutils import imSize, imreadRGB, imInvert, hStack
from trvo_utils.iter_utils import unzip
from trvo_utils.random import rnd0, rnd1

from utils.datasets import resize, to_yolo_input

excludeFonts = {
    0: ["AndaleMono", "Ayuthaya", "Krungthep", "Silom"]
}


class SyntheticNumberDataset(IterableDataset):
    class_names = [str(n) for n in range(10)]
    _emptyTargets = np.empty([0, 5], np.float32)
    maxNumberOfDigits = 9
    hPad, vPad, middlePad = 48, 48, 10
    padding = hPad, vPad, middlePad

    @staticmethod
    def _composeTransforms(transforms):
        bbox_params = BboxParams('pascal_voc', ['class_ids'], min_visibility=.5)
        return Compose(transforms or [], bbox_params)

    def __init__(self, dataset_dir, img_size, numOfItems=-1, excludeFonts=excludeFonts, transforms=None,
                 multiscale=True):
        self._digits = Digits(dataset_dir, excludeFonts)
        self.numOfItems = numOfItems
        self.img_size = img_size

        if transforms is None:
            self.transforms = None
        else:
            self.transforms = self._composeTransforms(transforms)

        self.multiscale = multiscale
        self.min_size = self.img_size - 3 * 32
        self.max_size = self.img_size + 3 * 32
        self.batch_count = 0

    def __iter__(self):
        if self.numOfItems is None or self.numOfItems <= 0:
            repeater = repeat(None)
        else:
            repeater = repeat(None, self.numOfItems)
        return (self.getItem() for _ in repeater)

    def _transformItem(self, image, vocBoxes, class_ids):
        if self.transforms:
            r = self.transforms(image=image, bboxes=vocBoxes, class_ids=class_ids)
            image = r['image']
            vocBoxes = r['bboxes']
            class_ids = r['class_ids']

        targets = self._emptyTargets
        if len(vocBoxes):
            yoloBoxes = BBox.voc2yolo_boxes(vocBoxes, imSize(image))
            targets = np.insert(yoloBoxes, 0, class_ids, 1)

        return image, targets, vocBoxes, class_ids

    def getItem(self, originalValues=False):
        numOfDigits = rnd0(self.maxNumberOfDigits)
        padding = rnd1(self.hPad), rnd1(self.vPad), rnd0(self.middlePad)

        numberImage, vocBoxes, class_ids = self._digits.randomNumber(numOfDigits, padding)

        numberImage, targets, vocBoxes, class_ids = self._transformItem(numberImage, vocBoxes, class_ids)
        if originalValues:
            return numberImage, vocBoxes, class_ids

        imageTensor, targetsTensor = to_yolo_input(numberImage, targets)
        return imageTensor, targetsTensor

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
            self.img_size = choice(range(self.min_size, self.max_size + 1, 32))
        # Resize images to input shape
        imgs = torch.stack([resize(img, self.img_size) for img in imgs])
        self.batch_count += 1
        return imgs, targets


class Digits:
    def __init__(self, datasetDir, exclude):
        self._allLabeledDigits = self._load(datasetDir, exclude)

    def randomNumber(self, numberOfDigits, padding):
        labeledDigits = choices(self._allLabeledDigits, k=numberOfDigits)
        labels, digits = unzip(labeledDigits, [], [])
        numberImage, boxes = hStack(digits, padding, fillValue=0)
        # boxes = [(x1, y1, x2, y2) for x1, y1, x2, y2 in boxes]
        return numberImage, boxes, labels

    @staticmethod
    def _load(datasetDir, exclude: dict):
        def readInverted(path):
            im = imreadRGB(path)
            return imInvert(im, im)

        def included(n, imgFile):
            excludedFonts = exclude.get(-1, []) + exclude.get(n, [])
            font = os.path.splitext(os.path.basename(imgFile))[0]
            excluded = font in excludedFonts
            return not excluded

        labeledDigits = []
        for n in range(10):
            numFiles = os.path.join(datasetDir, f'numeric_{n}', '*.png')
            numImages = [(n, readInverted(f)) for f in glob(numFiles) if included(n, f)]
            labeledDigits.extend(numImages)
        return labeledDigits
