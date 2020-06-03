import time
from glob import glob
import os
import cv2
from trvo_utils import toInt
from trvo_utils.imutils import imshowWait, imreadRGB

from my_src.YoloDetector import YoloDetector


def detect():
    class opt:
        model_def = "./data/yolov3.cfg"
        # weights = "./data/checkpoints/yolov3_ckpt_6_1.000.pth"
        weights = "./data/checkpoints/2/yolov3_ckpt_6.pth"
        class_path = "./data/classes.names"
        conf_thres = 0.5
        nms_thres = 0.5
        img_size = 416
        images = './data/counters/Musson_counters/train/*.jpg'  # './data/counters/4_from_phone/*.jpg'

    detector = YoloDetector(opt.model_def, "cuda", opt.img_size, opt.weights, opt.conf_thres, opt.nms_thres)

    img_files = sorted(glob(opt.images))
    colors = [(0, 0, 200), (200, 0, 0)]
    for img_file in img_files:
        img = imreadRGB(img_file)
        detections = detector(img)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
            color = colors[cls_pred]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        k = imshowWait(img=cv2.resize(img, None, None, .5, .5))
        if k == 27:
            break


if __name__ == '__main__':
    detect()
