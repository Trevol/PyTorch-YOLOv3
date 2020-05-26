import cv2

from my_src.my_utils.bbox import BBox


def readAnnotation(labelFileName):
    with open(labelFileName, 'rt') as f:
        line = f.readline()
        l, cx, cy, w, h = line.split(' ')
        return int(l), float(cx), float(cy), float(w), float(h)


def main():
    _, cx_norm, cy_norm, w_norm, h_norm = readAnnotation('dataset/labels/train.txt')
    img = cv2.imread('dataset/images/train.jpg')

    pt1, pt2 = BBox.yolo2voc(cx_norm, cy_norm, w_norm, h_norm, img.shape[:2])

    cv2.rectangle(img, pt1, pt2, (0, 0, 200))

    cv2.imshow("", img)
    cv2.waitKey()


main()
