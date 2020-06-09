import time
from glob import glob
import os
import cv2
from trvo_utils import toInt, glob_files
from trvo_utils.imutils import imshowWait, imreadRGB, rgb2bgr
from trvo_utils.timer import timeit

from my_src.YoloDetector import YoloDetector


class opt:
    model_def = "./data/yolov3.cfg"
    # weights = "./data/checkpoints/yolov3_ckpt_6_1.000.pth"
    weights = "./data/checkpoints/gpu_server/1/yolov3_ckpt_0_1.000.pt"
    # weights = 'test_counter.weights'
    class_path = "./data/classes.names"
    conf_thres = 0.8
    nms_thres = 0.5
    img_size = 416
    image_dirs = [
        # './data/counters/0_from_internet/all/*.jpg',
        # './data/counters/Musson_counters/val/*.jpg',
        # './data/counters/Musson_counters/train/*.jpg',

        './data/counters/4_from_phone/*.jpg',
        # './data/counters/3_from_phone/*.jpg',
        # './data/counters/1_from_phone/val/*.jpg'
    ]


def detect():
    device = "cuda"
    detector = YoloDetector(opt.model_def, device, opt.img_size, opt.weights, opt.conf_thres, opt.nms_thres)

    colors = [(0, 0, 200), (200, 0, 0)]
    for img_file in glob_files(opt.image_dirs):
        img = imreadRGB(img_file)
        detections = detector(img)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
            color = colors[cls_pred]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        k = imshowWait(
            img=[rgb2bgr(cv2.resize(img, None, None, .5, .5)), img_file]
        )
        if k == 27:
            break


def detect_and_save_screen():
    screensDir = './data/screens'
    os.makedirs(screensDir, exist_ok=True)

    detector = YoloDetector(opt.model_def, "cuda", opt.img_size, opt.weights, opt.conf_thres, opt.nms_thres)
    for i, img_file in enumerate(glob_files(opt.image_dirs)):
        img = imreadRGB(img_file)
        detections = detector(img)
        screensDetections = [toInt(x1, y1, x2, y2, cls_pred)
                             for x1, y1, x2, y2, conf, cls_conf, cls_pred
                             in detections
                             if cls_pred == 1]
        assert len(screensDetections) == 1, img_file

        for x1, y1, x2, y2, _ in screensDetections:
            screenImg = img[y1:y2, x1:x2]
            screenImg = rgb2bgr(screenImg)
            cv2.imwrite(os.path.join(screensDir, f'{i:06d}.png'), screenImg)


if __name__ == '__main__':
    # detect_and_save_screen()
    detect()
