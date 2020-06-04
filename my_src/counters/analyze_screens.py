from glob import glob

import cv2
from trvo_utils.imutils import imInvert, binarizeSauvola, imreadRGB, imshowWait


class _io:
    pad = 40

    def preprocess(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        gray = cv2.medianBlur(gray, 3, gray)
        inverted = imInvert(gray, out=gray)
        binarized = binarizeSauvola(inverted, windowSize=41, k=.1)
        binarized = imInvert(binarized, out=binarized)
        binarized = cv2.cvtColor(binarized, cv2.COLOR_GRAY2RGB)
        # imshowWait(DEBUG=binarized)
        return binarized

    def postprocess(self, boxes):
        pad = self.pad
        boxes = [(x1 - pad, y1 - pad, x2 - pad, y2 - pad) for x1, y1, x2, y2 in boxes]
        return boxes

    def extendDetectionBox(self, proposedBox):
        x1, y1, x2, y2 = proposedBox
        pad = self.pad
        return x1 - pad, y1 - pad, x2 + pad, y2 + pad


def main():
    paths = './data/screens/*.png'
    clahe = cv2.createCLAHE(clipLimit=40)
    for file in sorted(glob(paths)):
        img = imreadRGB(file)
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_eq_hist = cv2.equalizeHist(gray)
        gray_clahe = clahe.apply(gray)
        preprocessed = _io().preprocess(img)
        k = imshowWait(
            bgr=img[..., ::-1],
            preprocessed=preprocessed,
            gray=gray,
            gray_eq_hist=gray_eq_hist,
            gray_clahe=gray_clahe
        )
        if k == 27: break


main()
