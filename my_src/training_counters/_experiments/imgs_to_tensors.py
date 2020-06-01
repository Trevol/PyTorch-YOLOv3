import numpy as np
import torch
from torchvision.transforms import transforms
from trvo_utils.imutils import zeros, fill


def main():
    img1 = fill([10, 10, 3], 20)
    img2 = fill([10, 10, 3], 10)
    toTensor = transforms.ToTensor()
    batch = [toTensor(img1), toTensor(img2)]
    t = torch.stack(batch, 0, out=None)
    t2 = t.to('cuda')
    1

main()
