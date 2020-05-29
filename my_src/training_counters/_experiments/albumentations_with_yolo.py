import cv2
import numpy as np
from albumentations import ShiftScaleRotate, BboxParams, Compose, Rotate, Resize
from torchvision.transforms import Scale
from trvo_utils import toInt
from trvo_utils.imutils import fill, zeros, imshow, imshowWait, imSize
from trvo_utils.annotation.bbox import BBox

from my_src.training_counters import transforms

def drawYoloBoxes(image, yoloBoxes, color=(0, 0, 200)):
    for cx_norm, cy_norm, w_norm, h_norm in yoloBoxes:
        x1, y1, x2, y2 = BBox.yolo2voc(cx_norm, cy_norm, w_norm, h_norm, imSize(image))
        x1, y1, x2, y2 = toInt(x1, y1, x2, y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
    return image


def readAnnotatedImage():
    # 001007.txt
    # 0 0.5036411285400391 0.6073875144675925 0.37897618611653644 0.7852249710648148
    # 1 0.4399942715962728 0.3136466979980469 0.1735086441040039 0.0722677160192419

    imgFilename = '../counters/1_from_phone/train/001007.jpg'
    lblFilename = '../counters/1_from_phone/train/001007.txt'
    img = cv2.imread(imgFilename)
    annotations = np.loadtxt(lblFilename).reshape(-1, 5)
    boxes = annotations[:, 1:]
    class_ids = annotations[:, 0]
    return img, boxes, class_ids


def makeAnnotatedImage():
    img = zeros([300, 300, 3])
    y1, y2, x1, x2 = 100, 300, 100, 230
    img[y1:y2, x1:x2] = 255
    yoloBox = BBox.voc2yolo([x1, y1, x2, y2], imSize(img))
    return img, [yoloBox], [0]


def main():
    img, yoloBoxes, class_ids = readAnnotatedImage()

    # a = transforms.make(1)
    a = ShiftScaleRotate(shift_limit=0.1025, scale_limit=0.2, rotate_limit=15, interpolation=cv2.INTER_AREA,
                         border_mode=cv2.BORDER_CONSTANT, p=1)

    bboxParams = BboxParams('pascal_voc', label_fields=['class_ids'])
    a = Compose([], bboxParams)
    
    key = 0
    while key != 27:
        vocBoxes = BBox.yolo2voc_boxes(yoloBoxes, imSize(img))
        r = a(image=img, bboxes=vocBoxes, class_ids=class_ids)
        transformedImg = r["image"]
        transformedBoxes = BBox.voc2yolo_boxes(r['bboxes'], imSize(transformedImg))
        key = imshowWait(img=img, transformedImg=drawYoloBoxes(transformedImg, transformedBoxes))


main()
