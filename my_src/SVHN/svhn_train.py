from __future__ import division

import argparse
import math
from itertools import islice

from tqdm import tqdm

from models import Darknet
from my_src.SVHN import svhn_transforms
from my_src.utils.ModelMetrics import ModelMetrics
from my_src.utils.MultiDirDataset import MultiDirDataset
from my_src.utils.ModelEvaluator import ModelEvaluator
from utils.utils import *

import os

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR


class TrainingOptions:
    stages = 3
    epochsPerStage = 5  # number of epochs
    stepsPerEpoch = 10000
    initialLR = .001

    batch_size = 8  # size of each image batch
    gradient_accumulations = 2  # number of gradient accums before step
    model_def = "config/yolov3.cfg"  # path to model definition file
    pretrained_weights = ""  # if specified starts from checkpoint model
    checkpoints_path = ""
    n_cpu = 8  # number of cpu threads to use during batch generation
    img_size = 416  # size of each image dimension
    checkpoint_interval = 1  # interval between saving model weights
    evaluation_interval = 1  # interval evaluations on validation set
    compute_map = False  # if True computes mAP every tenth batch
    multiscale_training = True  # allow for multi-scale training
    trainDataDirs = []
    valDataDirs = []

    @classmethod
    def make(cls):
        parser = argparse.ArgumentParser()
        parser.add_argument("--batch_size", type=int, default=6, help="size of each image batch")
        parser.add_argument("--pretrained_weights", type=str, default="./data/weights/yolov3.weights",
                            help="if specified starts from checkpoint model")
        parser.add_argument("--checkpoints_path", type=str, default="./data/checkpoints",
                            help="where to store epoch checkpoints")
        args = parser.parse_args()

        opt = cls()

        opt.stages = 3
        opt.epochsPerStage = 5
        opt.stepsPerEpoch = 35000
        opt.initialLR = .001

        # opt.stages = 1
        # opt.epochsPerStage = 5
        # opt.stepsPerEpoch = 35000
        # opt.initialLR = .0001

        opt.batch_size = args.batch_size
        opt.pretrained_weights = args.pretrained_weights
        # opt.pretrained_weights = "./data/checkpoints/yolov3_ckpt_12_1.000.pth"
        opt.checkpoints_path = args.checkpoints_path

        opt.gradient_accumulations = 2
        opt.model_def = "./data/yolov3.cfg"

        opt.n_cpu = 0
        opt.img_size = 416
        opt.checkpoint_interval = 1
        opt.evaluation_interval = 1
        opt.compute_map = False
        opt.multiscale_training = True

        opt.trainDataDirs = [
            "./data/dataset/train"
        ]
        opt.valDataDirs = [
            "./data/dataset/test"
        ]
        return opt

    def makeModel(self, bindToDevice):
        model = Darknet(self.model_def).to(bindToDevice)
        model.apply(weights_init_normal)

        # If specified we start from checkpoint
        if self.pretrained_weights:
            if self.pretrained_weights.endswith(".pth"):
                model.load_state_dict(torch.load(self.pretrained_weights))
            else:
                model.load_darknet_weights(self.pretrained_weights)

        return model


def train():
    opt = TrainingOptions.make()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(opt.checkpoints_path, exist_ok=True)
    class_names = list(map(str, range(10)))

    model = opt.makeModel(device)

    # Get dataloader
    dataset = MultiDirDataset(opt.trainDataDirs, img_size=opt.img_size, label_names=class_names,
                              transforms=None,  # svhn_transforms.make(.5),
                              multiscale=opt.multiscale_training)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.initialLR)
    scheduler = StepLR(optimizer, step_size=opt.epochsPerStage, gamma=0.1)

    metrics = ModelMetrics()

    evalDataset = MultiDirDataset(opt.valDataDirs, opt.img_size, class_names, transforms=svhn_transforms.make(.5), multiscale=False)
    evalDataloader = DataLoader(
        evalDataset, batch_size=opt.batch_size, shuffle=False, num_workers=1,
        collate_fn=evalDataset.collate_fn
    )
    evaluator = ModelEvaluator(model, evalDataloader, opt.img_size, class_names, opt.batch_size,
                               saveToFile=os.path.join(opt.checkpoints_path, "yolov3_ckpt_{epoch}_eval.txt"))
    nBatches = 0
    print(opt.batch_size, opt.pretrained_weights, opt.checkpoints_path)
    for epoch in range(opt.stages * opt.epochsPerStage):
        model.train()

        if dataset.infinite:
            pbar = tqdm.tqdm(islice(dataloader, opt.stepsPerEpoch), total=opt.stepsPerEpoch)
        else:
            pbar = tqdm.tqdm(dataloader)

        pbar.set_description(f"Epoch {epoch}")
        lr = str(scheduler.get_last_lr()[0])
        pbar.set_postfix(loss="--", lr=lr)

        for imgs, targets in pbar:
            nBatches += 1

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()
            pbar.set_postfix(loss=loss.item(), lr=lr)
            if nBatches % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # metrics.log()

            model.seen += imgs.size(0)

        meanAP = math.nan
        if epoch % opt.evaluation_interval == 0:
            meanAP = evaluator.evaluateModel(epoch)

        if epoch % opt.checkpoint_interval == 0:
            checkpointPath = os.path.join(opt.checkpoints_path, f"yolov3_ckpt_{epoch}_{meanAP:.3f}.pth")
            torch.save(model.state_dict(), checkpointPath)

        scheduler.step()


train()
