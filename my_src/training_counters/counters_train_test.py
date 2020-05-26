from __future__ import division

from tqdm.gui import tqdm

from models import *
from my_src.training_counters.MultiDirDataset import MultiDirDataset
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *
from test import evaluate

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.optim as optim


class TrainingOptions:
    epochs = 100  # number of epochs
    batch_size = 8  # size of each image batch
    gradient_accumulations = 2  # number of gradient accums before step
    model_def = "config/yolov3.cfg"  # path to model definition file
    class_names = "classes.names"
    pretrained_weights = ""  # if specified starts from checkpoint model
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
        opt.batch_size = 6
        opt.gradient_accumulations = 2
        opt.model_def = "yolov3.cfg"
        opt.class_names = "classes.names"
        opt.pretrained_weights = "./weights/yolov3.weights"
        opt.n_cpu = 8
        opt.img_size = 416
        opt.checkpoint_interval = 1
        opt.evaluation_interval = 1
        opt.compute_map = False
        opt.multiscale_training = True

        opt.trainDataDirs = [
            "counters/1_from_phone/train",
            "counters/2_from_phone/train"
        ]
        opt.valDataDirs = [
            "counters/1_from_phone/val",
            "counters/2_from_phone/val"
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


class Metrics:
    # logger = Logger("logs")
    metrics = [
        "grid_size",
        "loss",
        "x",
        "y",
        "w",
        "h",
        "conf",
        "cls",
        "cls_acc",
        "recall50",
        "recall75",
        "precision",
        "conf_obj",
        "conf_noobj",
    ]

    def log(self, model, epoch, epochs, batch_i, len_dataloader, loss):
        # ----------------
        #   Log progress
        # ----------------

        log_str = "\n---- [Epoch %d/%d, Batch %d/%d] ----\n" % (epoch, epochs, batch_i, len_dataloader)

        metric_table = [["Metrics", *[f"YOLO Layer {i}" for i in range(len(model.yolo_layers))]]]

        # Log metrics at each YOLO layer
        for i, metric in enumerate(self.metrics):
            formats = {m: "%.6f" for m in self.metrics}
            formats["grid_size"] = "%2d"
            formats["cls_acc"] = "%.2f%%"
            row_metrics = [formats[metric] % yolo.metrics.get(metric, 0) for yolo in model.yolo_layers]
            metric_table += [[metric, *row_metrics]]

            # Tensorboard logging
            tensorboard_log = []
            for j, yolo in enumerate(model.yolo_layers):
                for name, metric in yolo.metrics.items():
                    if name != "grid_size":
                        tensorboard_log += [(f"{name}_{j + 1}", metric)]
            tensorboard_log += [("loss", loss.item())]
            # logger.list_of_scalars_summary(tensorboard_log, batches_done)

        log_str += AsciiTable(metric_table).table
        log_str += f"\nTotal loss {loss.item()}"

        # Determine approximate time left for epoch
        epoch_batches_left = len_dataloader - (batch_i + 1)
        time_left = datetime.timedelta(seconds=epoch_batches_left * (time.time() - start_time) / (batch_i + 1))
        log_str += f"\n---- ETA {time_left}"

        print(log_str)


def evaluateModel(model, valDataDirs, img_size, class_names):
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valid_path,
        iou_thres=0.5,
        conf_thres=0.5,
        nms_thres=0.5,
        img_size=img_size,
        batch_size=8,
    )
    evaluation_metrics = [
        ("val_precision", precision.mean()),
        ("val_recall", recall.mean()),
        ("val_mAP", AP.mean()),
        ("val_f1", f1.mean()),
    ]
    # logger.list_of_scalars_summary(evaluation_metrics, epoch)

    # Print class APs and mAP
    ap_table = [["Index", "Class name", "AP"]]
    for i, c in enumerate(ap_class):
        ap_table += [[c, class_names[c], "%.5f" % AP[i]]]
    print(AsciiTable(ap_table).table)
    print(f"---- mAP {AP.mean()}")


def train():
    opt = TrainingOptions.make()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs("checkpoints", exist_ok=True)
    class_names = load_classes(opt.class_names)

    # Initiate model
    model = opt.makeModel(device)

    # Get dataloader
    dataset = MultiDirDataset(opt.trainDataDirs, img_size=opt.img_size, augment=True,
                              multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn,
    )

    optimizer = torch.optim.Adam(model.parameters())

    metrics = Metrics()

    for epoch in range(opt.epochs):
        model.train()
        for batch_i, (_, imgs, targets) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            batches_done = len(dataloader) * epoch + batch_i

            imgs = Variable(imgs.to(device))
            targets = Variable(targets.to(device), requires_grad=False)

            loss, outputs = model(imgs, targets)
            loss.backward()

            if batches_done % opt.gradient_accumulations:
                # Accumulates gradient before each step
                optimizer.step()
                optimizer.zero_grad()

            # metrics.log()

            model.seen += imgs.size(0)

        if epoch % opt.evaluation_interval == 0:
            # evaluateModel()
            pass

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_{epoch}.pth")


train()
