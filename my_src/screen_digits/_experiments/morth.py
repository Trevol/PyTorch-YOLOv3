import imutils
import cv2
from trvo_utils.imutils import imshowWait


def main():
    img = cv2.imread('preprocessed_screen.png', cv2.IMREAD_GRAYSCALE)

    elem = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    img = cv2.erode(img, elem, iterations=1)

    imshowWait(img)
    # sk = imutils.skeletonize(img, (3, 3))
    # imshowWait(sk)


main()
