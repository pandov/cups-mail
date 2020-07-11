import torch
import torchvision.transforms.functional as TF
from torchvision import transforms

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

class RandomRotation(transforms.RandomRotation):
    def __call__(self, img, mask):
        angle = self.get_params(self.degrees)
        img = TF.rotate(img, angle, self.resample, self.expand, self.center, self.fill)
        mask = TF.rotate(mask, angle, self.resample, self.expand, self.center, self.fill)
        return img, mask

class RandomCrop(transforms.RandomCrop):
    def __call__(self, img, mask):
        if self.padding is not None:
            img = TF.pad(img, self.padding, self.fill, self.padding_mode)
            mask = TF.pad(mask, self.padding, self.fill, self.padding_mode)

        if self.pad_if_needed and img.size[0] < self.size[1] and mask.size[0] < self.size[1]:
            img = TF.pad(img, (self.size[1] - img.size[0], 0), self.fill, self.padding_mode)
            mask = TF.pad(mask, (self.size[1] - mask.size[0], 0), self.fill, self.padding_mode)

        if self.pad_if_needed and img.size[1] < self.size[0] and mask.size[1] < self.size[0]:
            img = TF.pad(img, (0, self.size[0] - img.size[1]), self.fill, self.padding_mode)
            mask = TF.pad(mask, (0, self.size[0] - mask.size[1]), self.fill, self.padding_mode)

        i, j, h, w = self.get_params(img, self.size)
        img = TF.crop(img, i, j, h, w)
        mask = TF.crop(mask, i, j, h, w)
        return img, mask

class Resize(transforms.Resize):
    def __call__(self, img, mask):
        img = TF.resize(img, self.size, self.interpolation)
        mask = TF.resize(mask, self.size, self.interpolation)
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
