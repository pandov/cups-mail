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

def get_stages_transform():
    return {
        'train': transforms.Compose([
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.2, hue=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation90(),
            transforms.RandomResizedCrop(size=(512, 640), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
            transforms.RandomGaussianBlur(1),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Negative(),
            transforms.Normalize(),
        ]),
        'valid': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Negative(),
            transforms.Normalize(),
        ]),
    }

class BACTERIA(ImageFolder):

    def __init__(self, stage, keys, apply_mask=False, **kwargs):
        kwargs['root'] = f'./dataset/processed/{stage}/samples'
        kwargs['loader'] = loader
        kwargs['transform'] = get_stages_transform().get(stage)
        super().__init__(**kwargs)
        self.keys = keys
        self.apply_mask = apply_mask

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        name = os.path.splitext(filepath)[0][-3:]
        image, mask = self.loader(filepath)
        image, mask = self.transform(image, mask)
        if self.apply_mask:
            image[:, mask[0] == 0] = 0
        
        items = {'name': name, 'image': image, 'mask': mask, 'label': label}
        return [items[key] for key in self.keys]

    def loader(self, **kwargs):
        return DataLoader(self, **kwargs)
