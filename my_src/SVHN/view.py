import os
import cv2
from glob import glob
from trvo_utils import toInt
from trvo_utils.annotation import PascalVocXmlParser
from trvo_utils.imutils import imshowWait, rgb2bgr, imreadRGB
from trvo_utils.timer import timeit

from my_src.SVHN import svhn_transforms
from my_src.SVHN.svhn_transforms import Invert
from my_src.utils.MultiDirDataset import MultiDirDataset
from tqdm import tqdm

def main():
    test_dir = ['data/dataset/train']
    labels = list(map(str, range(10)))
    with timeit():
        ds = MultiDirDataset(test_dir, 416, labels, svhn_transforms.make(.5))

    pbar = tqdm(ds.items_iter(originalValues=True))
    for imgFile, img, boxes, class_ids, index in pbar:
        pbar.set_postfix(index=index)
        # img = rgb2bgr(img, img)
        # for b in boxes:
        #     x1, y1, x2, y2 = toInt(*b)
        #     cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 200))
        # if imshowWait(img) == 27: break


def main_():
    inv = Invert(p=.8)
    allLabels = list(map(str, range(10)))
    with timeit():
        images = []
        annotations = []
        for f in glob('data/dataset/test/10312.png'):
            img = imreadRGB(f)
            images.append(img)
            annFile = os.path.splitext(f)[0] + '.xml'
            p = PascalVocXmlParser(annFile)
            boxes = p.boxes()
            labels = p.labels()
            class_ids = [allLabels.index(l) for l in labels]
            annotations.append((boxes, labels))
            while True:
                img = inv(image=img)["image"]
                imshowWait(rgb2bgr(img))


main()
