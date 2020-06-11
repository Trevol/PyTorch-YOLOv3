from glob import glob
import os
import cv2
from trvo_utils import toInt
from trvo_utils.imutils import imreadRGB, imshowWait, rgb2bgr

from my_src.YoloDetector import YoloDetector
from utils.sort_left2_right import sort_detections


def detect():
    model_def_file = 'data/yolov3.cfg'
    weights_file = 'data/checkpoints/gpu_server/1/yolov3_ckpt_0_0.634.pth'
    image_dir = 'data/dataset/test/*.png'

    detector = YoloDetector(model_def_file, "cuda", 416, weights_file, .8, .5)

    colors = [(0, 0, 200), (200, 0, 0)]
    for img_file in glob(image_dir):
        img = imreadRGB(img_file)
        detections = detector(img)
        detections = sort_detections(detections)
        print([int(d[6]) for d in detections])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
            # color = colors[cls_pred]
            color = colors[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        k = imshowWait(img=[rgb2bgr(img), img_file])
        if k == 27:
            break


def detect_screens():
    model_def_file = 'data/yolov3.cfg'
    weights_file = 'data/checkpoints/gpu_server/1/yolov3_ckpt_1_0.685.pth'
    image_dir = 'data/screens/musson/*.png'

    detector = YoloDetector(model_def_file, "cuda", 416, weights_file, .8, .5)

    colors = [(0, 0, 200), (200, 0, 0)]
    for img_file in sorted(glob(image_dir)):
        img = imreadRGB(img_file)
        detections = detector(img)
        detections = sort_detections(detections)
        print([int(d[6]) for d in detections])
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
            # color = colors[cls_pred]
            color = colors[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        k = imshowWait(img=[rgb2bgr(img), img_file])
        if k == 27:
            break


def detect_screens_side_by_side():
    model_def_file = 'data/yolov3.cfg'
    weights_files = ['data/checkpoints/gpu_server/1/yolov3_ckpt_1_0.685.pth',
                     'data/checkpoints/gpu_server/1/yolov3_ckpt_0_0.634.pth',
                     'data/checkpoints/gpu_server/2/yolov3_ckpt_0_0.686__.pth']
    image_dir = 'data/screens/musson/*.png'

    detectors = [
        [weights_file, YoloDetector(model_def_file, "cuda", 416, weights_file, .8, .5)]
        for weights_file in weights_files
    ]

    colors = [(0, 0, 200), (200, 0, 0)]
    for img_file in sorted(glob(image_dir)):
        img = imreadRGB(img_file)
        images = {}
        for weights_file, detector in detectors:
            img_copy = img.copy()
            detections = detector(img_copy)
            detections = sort_detections(detections)
            print([int(d[6]) for d in detections])
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
                # color = colors[cls_pred]
                color = colors[0]
                cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, 2)
            images[os.path.basename(weights_file)] = rgb2bgr(img_copy)
        k = imshowWait(**images)
        if k == 27:
            break


detect_screens_side_by_side()
