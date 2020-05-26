from glob import glob
import os
import cv2
from trvo_utils.imutils import imSize, imshowWait
from trvo_utils.voc_annotation import PascalVocXmlParser
from trvo_utils.voc_annotation.bbox import BBox

labelNames = ["counter", "counter_screen"]


def convertVocToYolo(vocFile):
    name, _ = os.path.splitext(os.path.basename(vocFile))
    parentDir, _ = os.path.split(vocFile)
    vocParser = PascalVocXmlParser(vocFile)
    vocBoxes, labels = vocParser.annotation()
    labels = [labelNames.index(l) for l in labels]

    yoloFile = os.path.join(parentDir, name + '.txt')
    imageFile = os.path.join(parentDir, os.path.basename(vocParser.filename()))
    img = cv2.imread(imageFile)
    imgSize = imSize(img)
    with open(yoloFile, 'wt') as f:
        for l, vocBox in zip(labels, vocBoxes):
            cxNorm, cyNorm, wNorm, hNorm = BBox.voc2yolo(vocBox, imgSize)
            f.write(f"{l} {cxNorm} {cyNorm} {wNorm} {hNorm}\n")


def main():
    dataDirs = [
        "counters/0_from_internet/all",
        "counters/0_from_internet/train",
        "counters/0_from_internet/val",

        "counters/1_from_phone/2_selected",
        "counters/1_from_phone/train",
        "counters/1_from_phone/val",

        "counters/2_from_phone/all",
        "counters/2_from_phone/train",
        "counters/2_from_phone/val"
    ]
    for dataDir in dataDirs:
        vocFiles = glob(os.path.join(dataDir, '*.xml'))
        for vocFile in vocFiles:
            convertVocToYolo(vocFile)


main()
