import torch

from models import Darknet
from my_src.training_counters.my_utils.ModelEvaluator import ModelEvaluator


def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Darknet('../yolov3.cfg').to(device)
    model.load_state_dict(torch.load("../checkpoints/2/yolov3_ckpt_6.pth"))

    valDataDirs = [
        # "../counters/1_from_phone/val",
        # "../counters/2_from_phone/val",
    ]
    class_names = ["counter", "counter_screen"]
    evaluator = ModelEvaluator(model, valDataDirs, 416, class_names, 6,
                               saveToFile="test_{epoch}_eval.txt")

    evaluator.evaluateModel(123)


test()
