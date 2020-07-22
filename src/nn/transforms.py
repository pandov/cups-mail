import torch
import torchvision.transforms.functional as TF
import cv2
import numpy as np
from torchvision import transforms
from PIL import Image, ImageFilter

class Compose(transforms.Compose):
    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

class Lambda(transforms.Lambda):
    def __call__(self, img, mask):
        return self.lambd(img), self.lambd(mask)

class RandomHorizontalFlip(transforms.RandomHorizontalFlip):
    def __call__(self, img, mask):
        if torch.rand(1) < self.p:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
        return img, mask

class RandomVerticalFlip(transforms.RandomVerticalFlip):
    def __call__(self, img, mask):
        if torch.rand(1) < self.p:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
        return img, mask

class RandomRotation(transforms.RandomRotation):
    def __call__(self, img, mask):
        angle = self.get_params(self.degrees)
        img = TF.rotate(img, angle, self.resample, self.expand, self.center, self.fill)
        mask = TF.rotate(mask, angle, self.resample, self.expand, self.center, self.fill)
        return img, mask

# class RandomRotation180(object):
#     def __call__(self, img, mask):
#         rot = torch.randint(0, 2, (1,))
#         angle = 180 * rot
#         img = TF.rotate(img, angle)
#         mask = TF.rotate(mask, angle)
#         return img, mask

class RandomRotation90(object):
    def __call__(self, img, mask):
        rot = torch.randint(0, 5, (1,))
        angle = 90 * rot
        img = TF.rotate(img, angle, False, True)
        mask = TF.rotate(mask, angle, False, True)
        return img, mask

# class RandomPerspective(transforms.RandomPerspective):
#     def __call__(self, img, mask):
#         if torch.rand(1) < self.p:
#             width, height = img.size
#             startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
#             img = TF.perspective(img, startpoints, endpoints, self.interpolation)
#             mask = TF.perspective(mask, startpoints, endpoints, self.interpolation)
#         return img, mask

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img, mask):
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        mask = TF.resized_crop(mask, i, j, h, w, self.size, self.interpolation)
        return img, mask

class ColorJitter(transforms.ColorJitter):
    def __call__(self, img, mask):
        fn_idx = torch.randperm(4)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = TF.adjust_brightness(img, brightness_factor)
                mask = TF.adjust_brightness(mask, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = TF.adjust_contrast(img, contrast_factor)
                mask = TF.adjust_contrast(mask, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = TF.adjust_saturation(img, saturation_factor)
                mask = TF.adjust_saturation(mask, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = TF.adjust_hue(img, hue_factor)
                mask = TF.adjust_hue(mask, hue_factor)

        return img, mask

# class RandomCrop(transforms.RandomCrop):
#     def __call__(self, img, mask):
#         i, j, h, w = self.get_params(img, self.size)
#         img = TF.crop(img, i, j, h, w)
#         mask = TF.crop(mask, i, j, h, w)
#         return img, mask

class Resize(transforms.Resize):
    def __call__(self, img, mask):
        img = TF.resize(img, self.size, self.interpolation)
        mask = TF.resize(mask, self.size, self.interpolation)
        return img, mask

class RandomGaussianBlur(object):
    def __init__(self, radius, p=0.5):
        self.radius = radius
        self.p = p

    def __call__(self, img, mask):
        if torch.rand(1) < self.p:
            radius = torch.rand(1) * self.radius
            blur = ImageFilter.GaussianBlur(radius)
            img = img.filter(blur)
        return img, mask

class Grayscale(transforms.Grayscale):
    def __call__(self, img, mask):
        img = TF.to_grayscale(img)
        mask = TF.to_grayscale(mask)
        return img, mask

# class Negative(transforms.Grayscale):
#     def __call__(self, img, mask):
#         img = 1 - img
#         return img, mask

class ToTensor(transforms.ToTensor):
    def __call__(self, img, mask):
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        return img, mask

class Normalize(object):

    def __call__(self, img, mask):
        img_np = np.asarray(img)

        dilated_img = cv2.dilate(img_np, np.ones((7, 7), np.uint8))
        bg_img = cv2.medianBlur(dilated_img, 21)
        diff_img = cv2.absdiff(img_np, bg_img)   
        norm_img = cv2.normalize(diff_img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)

        img = Image.fromarray(norm_img)

        # print(norm_img.shape)
        # img = torch.from_numpy(img_np).unsqueeze(0)
        # img /= 255
        return img, mask
