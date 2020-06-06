import cv2
from trvo_utils import toInt
from trvo_utils.imutils import imshowWait, imreadRGB, rgb2bgr, imInvert, binarizeSauvola, gray2bgr, imshow
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


def spatialSort_detections(yoloDetections):
    def x1(d):
        return d[0]

    return sorted(yoloDetections, key=x1)


def boxes_classIds(detections):
    boxes = []
    class_ids = []
    for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
        boxes.append((x1, y1, x2, y2))
        class_ids.append(int(cls_pred))
    return boxes, class_ids


class detect:
    def __init__(self):
        weights_file = './data/checkpoints/gpu_server/yolov3_ckpt_18_0.993.pth'
        self.detector = YoloDetector('./data/yolov3.cfg', 'cuda', 416, weights_file)

    def synthetic(self):
        dataset = SyntheticNumberDataset('./data/28x28', 416, transforms=transforms.make(1), multiscale=False)
        for i in range(1000):
            numberImage, vocBoxes, class_ids = dataset.getItem(originalValues=True)
            detections = self.detector(numberImage)
            detections = spatialSort_detections(detections)
            vocBoxes, class_ids = boxes_classIds(detections)

            numberImage = rgb2bgr(numberImage, numberImage)
            draw(numberImage, vocBoxes, class_ids)
            value = ''.join(map(str, class_ids))
            k = imshowWait([numberImage, value])
            if k == 27: break

    def synthetic_DEBUG(self):
        dataset = SyntheticNumberDataset('./data/28x28', 416, transforms=transforms.make(1), multiscale=False)
        for i in range(1000):
            numberImage, vocBoxes, class_ids = dataset.getItem(originalValues=True)
            detections = self.detector(numberImage)
            detections = spatialSort_detections(detections)

            numberImage = rgb2bgr(numberImage, numberImage)
            if self._DEBUG_detections(numberImage, detections) == 27: break

    def _DEBUG_detections(self, img, detections):
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
            img = cv2.rectangle(img.copy(), (x1, y1), (x2, y2), (0, 0, 200))

            title = f"{cls_pred}  {cls_conf.item():.3f}  {conf.item():.3f}"
            k = imshowWait(img=[img, title])
            if k == 27: return k
        return 0

    def screens(self):
        from glob import glob

        def preprocess(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 3, gray)
            inverted = imInvert(gray, out=gray)
            binarized = binarizeSauvola(inverted, windowSize=41, k=.1)
            binarized = imInvert(binarized, out=binarized)
            # imshowWait(DEBUG=binarized)
            return binarized

        elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        screens_dir = './data/screens/4/*.png'
        for f in sorted(glob(screens_dir)):
            img = imreadRGB(f)
            prepocessed = preprocess(img)

            prepocessed = cv2.erode(prepocessed, elem, iterations=1)

            detections = self.detector(prepocessed, .8)
            detections = spatialSort_detections(detections)
            vocBoxes, class_ids = boxes_classIds(detections)

            img = rgb2bgr(img, img)
            draw(img, vocBoxes, class_ids)
            value = ''.join(map(str, class_ids))
            k = imshowWait([img, value], prepocessed)
            if k == 27: break

    def screens_DEBUG(self):
        from glob import glob

        def preprocess(image):
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            gray = cv2.medianBlur(gray, 3, gray)
            inverted = imInvert(gray, out=gray)
            binarized = binarizeSauvola(inverted, windowSize=41, k=.1)
            binarized = imInvert(binarized, out=binarized)
            # imshowWait(DEBUG=binarized)
            return binarized

        elem = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        screens_dir = './data/screens/prepocessed_screenshot_06.06.2020.png'
        for f in sorted(glob(screens_dir)):
            img = imreadRGB(f)
            prepocessed = preprocess(img)

            # prepocessed = cv2.erode(prepocessed, elem, iterations=1)

            detections = self.detector(prepocessed, .6)
            detections = spatialSort_detections(detections)

            prepocessed = gray2bgr(prepocessed)

            imshow(prepocessed=prepocessed)
            if self._DEBUG_detections(prepocessed, detections) == 27: break


detect().screens_DEBUG()
# detect().synthetic()
