from my_src.utils.MultiDirDataset import MultiDirDataset


def main():
    test_dir = 'dataset/test'
    labels = list(range(10))
    ds = MultiDirDataset(test_dir, 416, labels)
    for i in ds:
        pass


main()
