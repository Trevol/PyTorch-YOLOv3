import cv2
from trvo_utils import toInt
from trvo_utils.imutils import imshowWait
from trvo_utils.iter_utils import unzip

from my_src.YoloDetector import YoloDetector
from my_src.screen_digits.SyntheticNumberDataset import SyntheticNumberDataset
from my_src.utils import transforms


def draw(img, boxes, class_ids):
    color = (0, 0, 200)
    for (x1, y1, x2, y2), class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = toInt(x1, y1, x2, y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color)
    return img


def spatialSort(boxes, class_ids):
    def box_x1(box_class_id):
        return box_class_id[0][0]

    ordered = sorted(zip(boxes, class_ids), key=box_x1)
    ordered_boxes, ordered_class_ids = unzip(ordered, [], [])
    return ordered_boxes, ordered_class_ids


def detect():
    detector = YoloDetector('./data/yolov3.cfg', 'cuda', 416, './data/checkpoints/gpu_server/yolov3_ckpt_19_0.732.pth')
    dataset = SyntheticNumberDataset(-1, './data/28x28', 416, transforms.make(1), False)
    for i in range(1000):
        numberImage, vocBoxes, class_ids = dataset.getItem(originalValues=True)
        numberImage = cv2.cvtColor(numberImage, cv2.COLOR_RGB2BGR, numberImage)
        vocBoxes, class_ids = spatialSort(vocBoxes, class_ids)
        value = ''.join(map(str, class_ids))
        draw(numberImage, vocBoxes, class_ids)
        k = imshowWait([numberImage, value])
        if k == 27: break


detect()
