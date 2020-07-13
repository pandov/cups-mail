import torch
import torchvision.transforms.functional as TF
from torchvision import transforms
from PIL import ImageFilter

class Compose(transforms.Compose):
    def __call__(self, img, mask):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask = t(img, mask)
        return img, mask

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

# class RandomRotation(transforms.RandomRotation):
#     def __call__(self, img, mask):
#         angle = self.get_params(self.degrees)
#         img = TF.rotate(img, angle, self.resample, self.expand, self.center, self.fill)
#         mask = TF.rotate(mask, angle, self.resample, self.expand, self.center, self.fill)
#         return img, mask

class RandomRotation180(object):
    def __call__(self, img, mask):
        rot = torch.randint(0, 2, (1,))
        angle = 180 * rot
        img = TF.rotate(img, angle)
        mask = TF.rotate(mask, angle)
        return img, mask

class RandomPerspective(transforms.RandomPerspective):
    def __call__(self, img, mask):
        if torch.rand(1) < self.p:
            width, height = img.size
            startpoints, endpoints = self.get_params(width, height, self.distortion_scale)
            img = TF.perspective(img, startpoints, endpoints, self.interpolation)
            mask = TF.perspective(mask, startpoints, endpoints, self.interpolation)
        return img, mask

class RandomResizedCrop(transforms.RandomResizedCrop):
    def __call__(self, img, mask):
        self.size = img.size[::-1]
        i, j, h, w = self.get_params(img, self.scale, self.ratio)
        img = TF.resized_crop(img, i, j, h, w, self.size, self.interpolation)
        mask = TF.resized_crop(mask, i, j, h, w, self.size, self.interpolation)
        return img, mask

# class RandomCrop(transforms.RandomCrop):
#     def __call__(self, img, mask):
#         i, j, h, w = self.get_params(img, self.size)
#         img = TF.crop(img, i, j, h, w)
#         mask = TF.crop(mask, i, j, h, w)
#         return img, mask

# class Resize(transforms.Resize):
#     def __call__(self, img, mask):
#         img = TF.resize(img, self.size, self.interpolation)
#         mask = TF.resize(mask, self.size, self.interpolation)
#         return img, mask

class RandomGaussianBlur(object):
    def __call__(self, img, mask):
        radius = torch.rand(1) * 1.5
        blur = ImageFilter.GaussianBlur(radius)
        img = img.filter(blur)
        return img, mask

class Grayscale(transforms.Grayscale):
    def __call__(self, img, mask):
        img = TF.to_grayscale(img, 3)
        mask = TF.to_grayscale(mask, 1)
        return img, mask

class ToTensor(transforms.ToTensor):
    def __call__(self, img, mask):
        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        return img, mask

class Normalize(transforms.Normalize):
    def __call__(self, img, mask):
        img = TF.normalize(img, self.mean, self.std, self.inplace)
        # mask = TF.normalize(mask, self.mean, self.std, self.inplace)
        return img, mask
