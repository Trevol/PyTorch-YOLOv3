import torch
import numpy as np
from torch.autograd import Variable
from torchvision.transforms import transforms
from trvo_utils.imutils import imSize

from models import Darknet
from utils.datasets import pad_to_square, resize
from utils.utils import non_max_suppression, rescale_boxes


class YoloDetector:
    toTensor = transforms.ToTensor()

    def __init__(self, model_def_file, device, input_size, weights_file, defaultConfThreshold=.8,
                 defaultNmsThreshold=.5):
        self.device = torch.device(device)
        self.model = Darknet(model_def_file, img_size=input_size).to(self.device)

        if weights_file.endswith(".weights"):
            self.model.load_darknet_weights(weights_file)
        else:
            self.model.load_state_dict(torch.load(weights_file))

        self.model.eval()

        self.defaultNmsThreshold = defaultNmsThreshold
        self.defaultConfThreshold = defaultConfThreshold

    def _preprocess(self, img):
        if len(img.shape) == 2:
            img = np.dstack([img, img, img])
        imgTensor = self.toTensor(img)
        imgTensor, _ = pad_to_square(imgTensor, 0)
        imgTensor = resize(imgTensor, self.model.img_size)
        return imgTensor

    def _inputBatch(self, imgs):
        batch = [self._preprocess(img) for img in imgs]
        batch = torch.stack(batch)
        if self.device.type == "cuda":
            batch = batch.to("cuda")
        return batch

    def detectOnBatch(self, imgs, confThreshold=None, nmsThreshold=None):
        assert len(imgs)
        confThreshold = confThreshold or self.defaultConfThreshold or .5
        nmsThreshold = nmsThreshold or self.defaultNmsThreshold or .4

        batch = self._inputBatch(imgs)
        input_var = Variable(batch)
        with torch.no_grad():
            detections = self.model(input_var)
        detections = non_max_suppression(detections, confThreshold, nmsThreshold)

        detections = [
            rescale_boxes(imgDetections, self.model.img_size, imSize(img))
            for imgDetections, img
            in zip(detections, imgs)
            # if imgDetections[6] >= confThreshold  # cls_pred>=confThreshold
        ]
        # x1, y1, x2, y2, conf, cls_conf, cls_pred
        return detections

    def detect(self, img, confThreshold=None, nmsThreshold=None):
        batch = [img]
        batchDetections = self.detectOnBatch(batch, confThreshold, nmsThreshold)
        detections = batchDetections[0]
        return detections

    def __call__(self, imgOrBatch, confThreshold=None, nmsThreshold=None):
        isBatch = isinstance(imgOrBatch, (list, tuple))
        if isBatch:
            return self.detectOnBatch(imgOrBatch, confThreshold, nmsThreshold)
        return self.detect(imgOrBatch, confThreshold, nmsThreshold)
