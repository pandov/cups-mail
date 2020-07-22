import os
import torch
from numpy.random import permutation
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.folder import default_loader
from src.nn import transforms

def loader(image_path):
    image = default_loader(image_path)
    mask_path = image_path.replace('samples', 'masks')
    mask = default_loader(mask_path)
    return image, mask

def resized(state):
    if state:
        return transforms.Resize((384, 480))
    return transforms.Lambda(lambda _: _)

def get_stages_transform(is_resized):
    return {
        'train': transforms.Compose([
            transforms.ColorJitter(saturation=0.25, hue=0.25),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation90(),
            transforms.RandomResizedCrop(size=(512, 640), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomGaussianBlur(1),
            transforms.Grayscale(),
            transforms.Normalize(),
            resized(is_resized),
            transforms.ToTensor(),
        ]),
        'valid': transforms.Compose([
            transforms.Grayscale(),
            transforms.Normalize(),
            resized(is_resized),
            transforms.ToTensor(),
        ]),
    }

class BACTERIA(ImageFolder):

    def __init__(self, stage, keys, is_resized=False, apply_mask=False, **kwargs):
        kwargs['root'] = f'./dataset/processed/{stage}/samples'
        kwargs['loader'] = loader
        kwargs['transform'] = get_stages_transform(is_resized).get(stage)
        super().__init__(**kwargs)
        self.keys = keys
        self.apply_mask = apply_mask

    def __getitem__(self, index):
        # index += 93
        filepath, label = self.samples[index]
        name = os.path.splitext(filepath)[0][-3:]
        image, mask = self.loader(filepath)
        image, mask = self.transform(image, mask)
        if self.apply_mask:
            image[:, mask[0] == 0] *= 0.5
        
        items = {'name': name, 'image': image, 'mask': mask, 'label': label}
        return [items[key] for key in self.keys]

    def data_loader(self, **kwargs):
        return DataLoader(self, **kwargs)
