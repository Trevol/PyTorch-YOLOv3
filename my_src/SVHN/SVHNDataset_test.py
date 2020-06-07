import os
import cv2
import json
from glob import glob

from trvo_utils.imutils import imshowWait, imInvert


class NumberAnnotation:
    class Box:
        __slots__ = ['height', 'label', 'left', 'top', 'width']

        def __init__(self, box: dict):
            self.left = int(box['left'])
            self.top = int(box['top'])
            self.width = int(box['width'])
            self.height = int(box['height'])
            self.label = int(box['label'])
            if self.label == 10:
                self.label = 0

        def pt1_pt2(self):
            return (self.left, self.top), (self.left + self.width, self.top + self.height)

    __slots__ = ['filename', 'boxes']

    def __init__(self, annotation: dict):
        self.filename = annotation['filename']
        self.boxes = [self.Box(b) for b in annotation['boxes']]

    @classmethod
    def fromJsonFile(cls, json_file):
        with open(json_file, 'rt') as f:
            digitStruct = json.load(f)
        return [cls(item) for item in digitStruct]


def main():
    dataset_dir = "dataset/test"
    annotations = NumberAnnotation.fromJsonFile(os.path.join(dataset_dir, 'digitStruct.json'))
    for item in annotations:
        imgPath = os.path.join(dataset_dir, item.filename)
        img = cv2.imread(imgPath)
        for b in item.boxes:
            pt1, pt2 = b.pt1_pt2()
            cv2.rectangle(img, pt1, pt2, (0, 0, 200))
        k = imshowWait(img)

        if k == 27: break


main()
