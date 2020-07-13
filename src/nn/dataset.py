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
    # assert image.size == mask.size, 'Size mismatch'
    return image, mask

def train_transform():
    return transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation180(),
        transforms.RandomPerspective(distortion_scale=0.05),
        transforms.RandomResizedCrop(size=(512, 640), scale=(0.75, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomGaussianBlur(),
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

def valid_transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

class KSubset(Subset):

    def __init__(self, dataset, indices, transform, target=None):
        super().__init__(dataset, indices)
        self.target = target
        self.transform = transform

    def __getitem__(self, idx):
        filename, image, mask, label = self.dataset[self.indices[idx]]
        image, mask = self.transform(image, mask)
        if self.target is None:
            return filename, image, mask, label
        else:
            returns = {'mask': mask, 'label': label}
            return image, returns[self.target]

class BACTERIA(ImageFolder):

    def __init__(self, target=None, **kwargs):
        kwargs['root'] = './dataset/processed/samples'
        kwargs['loader'] = loader
        super().__init__(**kwargs)
        self.target = target

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        filename = os.path.splitext(filepath)[0][-3:]
        image, mask = self.loader(filepath)
        return filename, image, mask, label

    def crossval(self, kfold, batch_size=None):
        length = len(self)
        size = length // kfold
        idx = permutation(length).tolist()
        for i in range(kfold):
            start = size * i
            end = start + size
            train_idx = idx[:start] + idx[end:]
            valid_idx = idx[start:end]
            train_dataset = KSubset(self, train_idx, transform=train_transform(), target=self.target)
            valid_dataset = KSubset(self, valid_idx, transform=valid_transform(), target=self.target)
            if batch_size is None:
                datasets = {'train': train_dataset, 'valid': valid_dataset}
                yield datasets
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
                loaders = {'train': train_dataloader, 'valid': valid_dataloader}
                yield loaders
