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
            # transforms.Negative(),
            transforms.Normalize(),
        ]),
        'valid': transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
            # transforms.Negative(),
            transforms.Normalize(),
        ]),
    }

class StageSubset(Subset):

    def __init__(self, dataset, indices, stage):
        super().__init__(dataset, indices)
        self.stage = stage

    def __getitem__(self, idx):
        return self.dataset.getitem(self.indices[idx], self.stage)

class BACTERIA(ImageFolder):

    def __init__(self, keys, apply_mask=False, **kwargs):
        kwargs['root'] = './dataset/processed/samples'
        kwargs['loader'] = loader
        kwargs['transform'] = get_stages_transform()
        super().__init__(**kwargs)
        self.keys = keys
        self.apply_mask = apply_mask

    def getitem(self, index, stage='train'):
        filepath, label = self.samples[index]
        name = os.path.splitext(filepath)[0][-3:]
        image, mask = self.loader(filepath)
        if stage is not None:
            image, mask = self.transform[stage](image, mask)
            if self.apply_mask:
                image[:, mask[0] == 0] = 0
        
        items = {'name': name, 'image': image, 'mask': mask, 'label': label}
        return [items[key] for key in self.keys]

    def __getitem__(self, index):
        return self.getitem(index)

    def crossval(self, kfold, batch_size=None):
        length = len(self)
        size = length // kfold
        idx = permutation(length).tolist()
        for i in range(kfold):
            start = size * i
            end = start + size
            train_idx = idx[:start] + idx[end:]
            valid_idx = idx[start:end]
            train_dataset = StageSubset(self, train_idx, stage='train')
            valid_dataset = StageSubset(self, valid_idx, stage='valid')
            if batch_size is None:
                datasets = {'train': train_dataset, 'valid': valid_dataset}
                yield datasets
            else:
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
                valid_batch_size = min(batch_size, len(valid_dataset))
                valid_dataloader = DataLoader(valid_dataset, batch_size=valid_batch_size)
                loaders = {'train': train_dataloader, 'valid': valid_dataloader}
                yield loaders
