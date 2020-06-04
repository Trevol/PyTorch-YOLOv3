from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose, ISONoise, RGBShift
)
import cv2


def make(p=0.5):
    return Compose([
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
            ISONoise()
        ], p=0.9),
        MotionBlur(p=0.3),
        ShiftScaleRotate(shift_limit=0.0925, scale_limit=0.4, rotate_limit=7, border_mode=cv2.BORDER_CONSTANT,
                         value=0, p=0.6),
        # IAAPerspective(scale=(.055, .060), keep_size=False, p=.2),
        # OpticalDistortion(p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
        RGBShift(40, 40, 40)
    ], p=p)
