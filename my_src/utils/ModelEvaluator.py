import numpy as np
import torch
from terminaltables import AsciiTable
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.utils import xywh2xyxy, non_max_suppression, get_batch_statistics, ap_per_class


class ModelEvaluator:
    FloatTensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    def __init__(self, model, dataloader, img_size, class_names, batch_size, saveToFile):
        self.saveToFile = saveToFile
        self.batch_size = batch_size
        self.class_names = class_names
        self.img_size = img_size
        self.model = model
        self.dataloader = dataloader

    @staticmethod
    def _evaluate(model, dataloader, iou_thres, conf_thres, nms_thres, img_size):
        model.eval()

        labels = []
        sample_metrics = []  # List of tuples (TP, confs, pred)
        for imgs, targets in tqdm(dataloader, desc="Evaluating"):
            # Extract labels
            labels += targets[:, 1].tolist()
            # Rescale target
            targets[:, 2:] = xywh2xyxy(targets[:, 2:])
            targets[:, 2:] *= img_size

            imgs = Variable(imgs.type(ModelEvaluator.FloatTensor), requires_grad=False)

            with torch.no_grad():
                outputs = model(imgs)
                outputs = non_max_suppression(outputs, conf_thres=conf_thres, nms_thres=nms_thres)

            sample_metrics += get_batch_statistics(outputs, targets, iou_threshold=iou_thres)

        # Concatenate sample statistics
        true_positives, pred_scores, pred_labels = [np.concatenate(x, 0) for x in list(zip(*sample_metrics))]
        precision, recall, AP, f1, ap_class = ap_per_class(true_positives, pred_scores, pred_labels, labels)

        return precision, recall, AP, f1, ap_class

    def evaluateModel(self, epoch):
        # Evaluate the model on the validation set
        precision, recall, AP, f1, ap_class = self._evaluate(
            self.model,
            dataloader=self.dataloader,
            iou_thres=0.5,
            conf_thres=0.5,
            nms_thres=0.5,
            img_size=self.img_size
        )
        meanAP = AP.mean()
        evaluation_metrics = [
            ("val_precision", precision.mean()),
            ("val_recall", recall.mean()),
            ("val_mAP", meanAP),
            ("val_f1", f1.mean()),
        ]
        # logger.list_of_scalars_summary(evaluation_metrics, epoch)

        # Print class APs and mAP
        ap_table = [["Index", "Class name", "AP"]]
        for i, c in enumerate(ap_class):
            ap_table += [[c, self.class_names[c], "%.5f" % AP[i]]]

        self.report(epoch, AsciiTable(ap_table).table, meanAP)
        return meanAP

    def report(self, epoch, table, apMean):
        if self.saveToFile:
            file = self.saveToFile.format(epoch=epoch)
            with open(file, "wt") as f:
                f.write(f"Epoch {epoch}")
                f.write("\n")
                f.write(table)
                f.write("\n")
                f.write(f"---- mAP {apMean}")
        else:
            print(f"Epoch {epoch}")
            print(table)
            print(f"---- mAP {apMean}")
