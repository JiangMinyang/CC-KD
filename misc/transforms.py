import numbers
import random
import numpy as np
from PIL import Image, ImageOps, ImageFilter
# from config import cfg
import torch
import cv2
# ===============================img tranforms============================

class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask):
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class RandomHorizontallyFlip(object):
    def __call__(self, image, labels):
        height, width, _ = image.shape
        if random.random() > 0.5:
            if (len(labels) > 0):
                labels[:, 0] = width - labels[:, 0]
            image = np.fliplr(image)
        return image, labels

class HSV(object):
    """docstring for HSV"""
    def __init__(self, fraction=0.5):
        super(HSV, self).__init__()
        self.fraction = fraction

    def __call__(self, image, labels):
        # SV augmentation by 50%
        a_S = (random.random() * 2 - 1) * self.fraction + 1
        a_V = (random.random() * 2 - 1) * self.fraction + 1
        img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        S = img_hsv[:, :, 1].astype(np.float32)
        V = img_hsv[:, :, 2].astype(np.float32)

        S *= a_S
        if a_S > 1:
            np.clip(S, a_min=0, a_max=255, out=S)

        V *= a_V
        if a_V > 1:
            np.clip(V, a_min=0, a_max=255, out=V)

        img_hsv[:, :, 1] = S.astype(np.uint8)
        img_hsv[:, :, 2] = V.astype(np.uint8)
        image = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR)

        return image, labels
        

class Scale(object):
    def __init__(self, height, width, color=(127.5, 127.5, 127.5)):
        self.height = height
        self.width = width
        self.bg_color = color
    def __call__(self, image, labels):
        h, w, _ = image.shape
        img, ratio, padw, padh = self._letterbox(image, self.height, self.width, self.bg_color)

        if (len(labels) > 0):
            labels[:, 0] = ratio * w * labels[:, 0] + padw
            labels[:, 1] = ratio * h * labels[:, 1] + padh

        return img, labels

    def _letterbox(self, img, height, width, color, interpolation=cv2.INTER_LINEAR):  # resize a rectangular image to a padded rectangular 
        shape = img.shape[:2]  # shape = [height, width]
        ratio = min(float(height)/shape[0], float(width)/shape[1])
        new_shape = (round(shape[1] * ratio), round(shape[0] * ratio)) # new_shape = [width, height]
        dw = (width - new_shape[0]) / 2  # width padding
        dh = (height - new_shape[1]) / 2  # height padding
        top, bottom = round(dh - 0.1), round(dh + 0.1)
        left, right = round(dw - 0.1), round(dw + 0.1)
        img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
        return img, ratio, dw, dh

class RandomCrop(object):
    def __init__(self, low_ratio = 4, high_ratio = 1):
        self.low = low_ratio
        self.high = high_ratio

    def __call__(self, img, labels):
        h, w, _ = img.shape
        width = random.randint(w // self.low, w // self.high)
        height = random.randint(h // self.low, h // self.high)
        w1 = random.randint(0, w - width)
        h1 = random.randint(0, h - height)


        if (len(labels) > 0):
            labels[:, 0] -= w1 / w
            labels[:, 1] -= h1 / h

            labels = np.array([[pos[0] * w / width, pos[1] * h / height] for pos in labels if pos[0] >= 0 and pos[0] < width / w and pos[1] >= 0 and pos[1] < height / h])

        return img[h1: h1 + height, w1: w1 + width], labels

class Basic(object):
    def __init__(self):
        pass

    def __call__(self, img, labels):
        h, w, _ = img.shape
        labels[:, 0] *= w
        labels[:, 1] *= h
        return img, labels


class RandomAffine(object):
    def __init__(self, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), borderValue=(127.5, 127.5, 127.5)):
        self.degrees = degrees
        self.translate = translate
        self.scale = scale
        self.borderValue = borderValue

    def __call__(self, image, labels):
        height, width = image.shape[:2]
        M, a = self._gen_transform_param((width, height), degrees=self.degrees, translate=self.translate, scale=self.scale)
        img = cv2.warpPerspective(image, M, dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=self.borderValue)

        # Return warped points also
        if len(labels) > 0:
            n = labels.shape[0]
            points = labels.copy()

            # warp points
            xy = np.ones((n, 3))
            xy[:, :2] = points[:, [0, 1]].reshape(n, 2)  # xy1
            xy = (xy @ M.T)[:, :2].reshape(n, 2)

            xy =[x for x in xy if (x[0] >=0 and x[0] <= width - 1 and x[1] >= 0 and x[1] <= height - 1)]

            labels = np.array(xy)
        return img, labels


    def _gen_transform_param(self, shape, degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-2, 2)):
        # torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
        # https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
        border = 0  # width of added border (optional)

        # Rotation and Scale
        R = np.eye(3)
        a = random.random() * (degrees[1] - degrees[0]) + degrees[0]
        # a += random.choice([-180, -90, 0, 90])  # 90deg rotations added to small rotations
        s = random.random() * (scale[1] - scale[0]) + scale[0]
        R[:2] = cv2.getRotationMatrix2D(angle=a, center=(shape[0] / 2.0, shape[1] / 2.0), scale=s)

        # Translation
        T = np.eye(3)
        T[0, 2] = (random.random() * 2 - 1) * translate[0] * shape[0] + border  # x translation (pixels)
        T[1, 2] = (random.random() * 2 - 1) * translate[1] * shape[1] + border  # y translation (pixels)

        # Shear
        # S = np.eye(3)
        # S[0, 1] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # x shear (deg)
        # S[1, 0] = math.tan((random.random() * (shear[1] - shear[0]) + shear[0]) * math.pi / 180)  # y shear (deg)

        # M = S @ T @ R  # Combined rotation matrix. ORDER IS IMPORTANT HERE!!
        M = T @ R
        return M, a

class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class MaskToTensor(object):
    def __call__(self, img):
        return torch.from_numpy(np.array(img, dtype=np.int32)).long()


class LabelScale(object):
    def __init__(self, para):
        self.para = para

    def __call__(self, hm):
        tensor = torch.from_numpy(hm * self.para)
        return tensor

class GTScaleDown(object):
    def __init__(self, factor=8):
        self.factor = factor

    def __call__(self, img):
        w, h = img.shape[:2]
        if self.factor==1:
            return img

        img = cv2.resize(img, (h//self.factor, w//self.factor), interpolation=cv2.INTER_AREA) * self.factor * self.factor
        return img

