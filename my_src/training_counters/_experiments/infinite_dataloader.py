from itertools import repeat, chain, cycle, islice

from my_src.training_counters import transforms
from my_src.training_counters.my_utils.MultiDirDataset import MultiDirDataset
from torch.utils.data import DataLoader

from utils.utils import load_classes


def main():
    trainDataDirs = [
        # "../counters/1_from_phone/train",
        "../counters/2_from_phone/train"
    ]

    batchSize = 7
    class_names = load_classes('../classes.names')
    dataset = MultiDirDataset(trainDataDirs, 416, class_names,
                              transforms= transforms.make(1),
                              multiscale=False)
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    for imgs, targets in islice(dataloader, 100):
        print(targets)



main()
