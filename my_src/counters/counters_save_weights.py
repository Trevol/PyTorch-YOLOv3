import cv2
import torch
from trvo_utils import glob_files, toInt
from trvo_utils.imutils import imreadRGB

from models import Darknet
from my_src.YoloDetector import YoloDetector


def main():
    model_def_file = 'data/yolov3.cfg'
    weights = "./data/checkpoints/gpu_server/1/yolov3_ckpt_0_1.000.pt"
    model = Darknet(model_def_file, img_size=416)
    model.load_state_dict(torch.load(weights))
    model.save_darknet_weights('test_counter.weights')


def detect():
    class opt:
        model_def = "./data/yolov3.cfg"
        # weights = "./data/checkpoints/yolov3_ckpt_6_1.000.pth"
        weights = "./data/checkpoints/gpu_server/1/yolov3_ckpt_0_1.000.pt"
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

    detector = YoloDetector(opt.model_def, "cuda", opt.img_size, opt.weights, opt.conf_thres, opt.nms_thres)

    colors = [(0, 0, 200), (200, 0, 0)]
    for img_file in glob_files(opt.image_dirs):
        img = imreadRGB(img_file)
        detections = detector(img)

        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
            color = colors[cls_pred]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        k = imshowWait(
            img=[cv2.resize(img, None, None, .5, .5)[..., ::-1], img_file]
        )
        if k == 27:
            break


main()
