import torch


def main():
    t = torch.BoolTensor(2, 2).fill_(1)
    n = t.cpu().numpy()
    print(n)
    # print(t)


main()
