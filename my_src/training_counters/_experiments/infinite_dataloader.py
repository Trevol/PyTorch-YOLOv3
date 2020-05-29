from itertools import repeat, chain, cycle, islice

from my_src.training_counters import transforms
from my_src.training_counters.MultiDirDataset import MultiDirDataset
from torch.utils.data import DataLoader


def main():
    trainDataDirs = [
        "../counters/1_from_phone/train",
        "../counters/2_from_phone/train"
    ]

    batchSize = 7
    dataset = MultiDirDataset(trainDataDirs, img_size=416, transforms=transforms.make(1),
                              multiscale=True)
    dataloader = DataLoader(
        dataset,
        batch_size=batchSize,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        collate_fn=dataset.collate_fn
    )

    j = 0

    for i, imgs, targets in islice(cycle(dataloader), 4):
        print(j, len(imgs))
        j += 1
        if j > 300:
            break


main()
