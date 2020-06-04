from __future__ import division

import math
from itertools import islice

from tqdm import tqdm

from models import Darknet
from my_src.utils.ModelMetrics import ModelMetrics
from my_src.utils.MultiDirDataset import MultiDirDataset
from my_src.utils.ModelEvaluator import ModelEvaluator
from utils.utils import *
from utils.datasets import *

from terminaltables import AsciiTable

import os

import torch
from torch.utils.data import DataLoader
from torch.autograd import Variable

from my_src.utils import transforms


class TrainingOptions:
    epochs = 100  # number of epochs
    stepsPerEpoch = 1000
    batch_size = 8  # size of each image batch
    lr = 1e-4
    gradient_accumulations = 2  # number of gradient accums before step
    model_def = "config/yolov3.cfg"  # path to model definition file
    class_names = "classes.names"
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
        opt = cls()

        opt.epochs = 20
        opt.stepsPerEpoch = 1000
        opt.batch_size = 8
        opt.lr = 1e-5
        opt.gradient_accumulations = 2
        opt.model_def = "./data/yolov3.cfg"
        opt.class_names = "./data/classes.names"
        # opt.pretrained_weights = "./data/weights/yolov3.weights"
        opt.pretrained_weights = "./data/checkpoints/yolov3_ckpt_12_1.000.pth"

        opt.checkpoints_path = "./data/checkpoints"
        opt.n_cpu = 8
        opt.img_size = 416
        opt.checkpoint_interval = 1
        opt.evaluation_interval = 1
        opt.compute_map = False
        opt.multiscale_training = True

        opt.trainDataDirs = [
            "./data/counters/1_from_phone/train",
            "./data/counters/2_from_phone/train",
            "./data/counters/Musson_counters/train"
        ]
        opt.valDataDirs = [
            "./data/counters/1_from_phone/val",
            "./data/counters/2_from_phone/val",
            "./data/counters/Musson_counters/val"
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

    # os.makedirs(opt.checkpoints_path, exist_ok=True)
    class_names = load_classes(opt.class_names)

    model = opt.makeModel(device)

    # Get dataloader
    dataset = MultiDirDataset(opt.trainDataDirs, img_size=opt.img_size, label_names=class_names,
                              transforms=transforms.make(1),
                              multiscale=opt.multiscale_training)
    dataloader = DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)

    metrics = ModelMetrics()

    evalDataset = MultiDirDataset(opt.valDataDirs, opt.img_size, class_names, transforms=None, multiscale=False)
    evalDataloader = DataLoader(
        evalDataset, batch_size=opt.batch_size, shuffle=False, num_workers=1,
        collate_fn=evalDataset.collate_fn
    )
    evaluator = ModelEvaluator(model, evalDataloader, opt.img_size, class_names, opt.batch_size,
                               saveToFile=os.path.join(opt.checkpoints_path, "yolov3_ckpt_{epoch}_eval.txt"))
    nBatches = 0
    for epoch in range(opt.epochs):
        model.train()

        pbar = tqdm.tqdm(islice(dataloader, opt.stepsPerEpoch), total=opt.stepsPerEpoch)
        pbar.set_description(f"Epoch {epoch}")
        pbar.set_postfix(loss="--")

        for imgs, targets in pbar:
            nBatches += 1

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()
            pbar.set_postfix(loss=loss.item())
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


train()
