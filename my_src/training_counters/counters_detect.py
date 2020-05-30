import time

import cv2
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from trvo_utils import toInt
from trvo_utils.imutils import imSize, imshowWait

from models import Darknet
from utils.datasets import ImageFolder
from utils.utils import load_classes, non_max_suppression, rescale_boxes


def detect():
    class OPT:
        pass

    opt = OPT()
    opt.image_folder = './counters/Счетчики'
    opt.model_def = "yolov3.cfg"
    opt.weights_path = "./checkpoints/2/yolov3_ckpt_6.pth"
    opt.class_path = "classes.names"
    opt.conf_thres = 0.5
    opt.nms_thres = 0.3
    opt.batch_size = 1
    opt.n_cpu = 0
    opt.img_size = 416
    opt.checkpoint_model = ""

    deviceType = "cuda"
    device = torch.device(deviceType)

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
        img = cv2.imread(path)
        if detections is not None:
            # Rescale boxes to original image
            detections = rescale_boxes(detections, opt.img_size, imSize(img))
            unique_labels = detections[:, -1].cpu().unique()
            bbox_colors = colors
            for x1, y1, x2, y2, conf, cls_conf, cls_pred in detections:
                x1, y1, x2, y2, cls_pred = toInt(x1, y1, x2, y2, cls_pred)
                color = bbox_colors[cls_pred]
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        k = imshowWait(img=cv2.resize(img, None, None, .5, .5))
        if k == 27:
            break


if __name__ == '__main__':
    detect()
