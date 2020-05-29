from __future__ import division

from itertools import islice

import cv2
from tqdm import tqdm
from trvo_utils import toInt
from trvo_utils.imutils import imreadRGB, imSize, imshowWait

from models import *
from my_src.training_counters.MultiDirDataset import MultiDirDataset
from utils.utils import *
from utils.datasets import *
from utils.parse_config import *

from terminaltables import AsciiTable

import os
import sys
import time
import datetime
import argparse

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable
import torch.optim as optim

from my_src.training_counters import transforms


class TrainingOptions:
    epochs = 100  # number of epochs
    stepsPerEpoch = 1000
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

        opt.epochs = 30
        opt.stepsPerEpoch = 1000
        opt.batch_size = 6
        opt.gradient_accumulations = 2
        opt.model_def = "yolov3.cfg"
        opt.class_names = "classes.names"
        opt.pretrained_weights = "./checkpoints/1/yolov3_ckpt_3.pth"
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

        print(log_str)


def evaluate(model, path, iou_thres, conf_thres, nms_thres, img_size, batch_size):
    model.eval()

    # Get dataloader
    dataset = ListDataset(path, img_size=img_size, augment=False, multiscale=False)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=1, collate_fn=dataset.collate_fn
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    labels = []
    sample_metrics = []  # List of tuples (TP, confs, pred)
    for batch_i, (_, imgs, targets) in enumerate(tqdm.tqdm(dataloader, desc="Detecting objects")):
        # Extract labels
        labels += targets[:, 1].tolist()
        # Rescale target
        targets[:, 2:] = xywh2xyxy(targets[:, 2:])
        targets[:, 2:] *= img_size

        imgs = Variable(imgs.type(Tensor), requires_grad=False)

        with torch.no_grad():
            outputs = model(imgs)
            outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

        sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

    # Concatenate sample statistics
    true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
    precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

    return precision, recall, AP, f1, ap_class


def evaluateModel(model, valDataDirs, img_size, class_names):
    print("\n---- Evaluating Model ----")
    # Evaluate the model on the validation set
    precision, recall, AP, f1, ap_class = evaluate(
        model,
        path=valDataDirs,
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
    dataset = MultiDirDataset(opt.trainDataDirs, img_size=opt.img_size, label_names=class_names,
                              transforms=transforms.make(1),
                              multiscale=opt.multiscale_training)
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
        pin_memory=True,
        collate_fn=dataset.collate_fn
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    metrics = Metrics()

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

        if epoch % opt.evaluation_interval == 0:
            # evaluateModel()
            pass

        if epoch % opt.checkpoint_interval == 0:
            torch.save(model.state_dict(), f"checkpoints/yolov3_ckpt_{epoch}.pth")


def detect():
    class OPT:
        pass

    opt = OPT()
    opt.image_folder = './counters/5_from_phone'
    opt.model_def = "yolov3.cfg"
    opt.weights_path = "./checkpoints/yolov3_ckpt_6.pth"
    opt.class_path = "classes.names"
    opt.conf_thres = 0.8
    opt.nms_thres = 0.3
    opt.batch_size = 1
    opt.n_cpu = 0
    opt.img_size = 416
    opt.checkpoint_model = ""
    opt.outputDir = "./output"

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    deviceType = "cuda"
    device = torch.device(deviceType)

    os.makedirs(opt.outputDir, exist_ok=True)

    # Set up model
    model = Darknet(opt.model_def, img_size=opt.img_size).to(device)
    model.load_state_dict(torch.load(opt.weights_path))

    model.eval()  # Set in evaluation mode

    dataloader = DataLoader(
        ImageFolder(opt.image_folder, img_size=opt.img_size),
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=opt.n_cpu,
    )

    classes = load_classes(opt.class_path)  # Extracts class labels from file

    Tensor = torch.cuda.FloatTensor if deviceType == "cuda" else torch.FloatTensor

    imgs = []  # Stores image paths
    img_detections = []  # Stores detections for each image index

    print("\nPerforming object detection:")

    for img_paths, input_imgs in dataloader:
        prev_time = time.time()
        # Configure input
        input_imgs = Variable(input_imgs.type(Tensor))

        # Get detections
        with torch.no_grad():
            detections = model(input_imgs)
            detections = non_max_suppression(detections, opt.conf_thres, opt.nms_thres)

        # Log progress
        current_time = time.time()
        inference_time = current_time - prev_time
        print(f"\t+ Inference Time: {inference_time}")

        # Save image and detections
        imgs.extend(img_paths)
        img_detections.extend(detections)

    colors = [(0, 0, 200), (200, 0, 0)]

    # Iterate through images and save plot of detections
    for path, detections in zip(imgs, img_detections):
        if detections is None:
            continue
        img = cv2.imread(path)
        # Rescale boxes to original image
        detections = rescale_boxes(detections, opt.img_size, imSize(img))
        unique_labels = detections[:, -1].cpu().unique()
        n_cls_preds = len(unique_labels)
        bbox_colors = colors
        for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
            x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
            color = bbox_colors[cls_pred]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        k = imshowWait(img=cv2.resize(img, None, None, .5, .5))
        if k == 27:
            break


detect()
# train()
