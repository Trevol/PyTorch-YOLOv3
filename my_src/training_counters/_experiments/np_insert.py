import numpy as np


def main():
    boxes = np.zeros([0, 4]).tolist()
    class_ids = []
    targets = np.insert(boxes, 0, class_ids, 1)

    print(targets)

main()
