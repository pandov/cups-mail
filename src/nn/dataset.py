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

def transform():
    return transforms.Compose([
        transforms.Grayscale(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

class CrossValidationDataset(object):

    def crossval(self, kfold, batch_size):
        length = len(self)
        size = length // kfold
        idx = permutation(length).tolist()
        for i in range(kfold):
            start = size * i
            end = start + size
            train_idx = idx[:start] + idx[end:]
            valid_idx = idx[start:end]
            train_dataset = Subset(self, train_idx)
            valid_dataset = Subset(self, valid_idx)
            train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
            valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size)
            loaders = {'train': train_dataloader, 'valid': valid_dataloader}
            yield loaders

class BACTERIA(ImageFolder, CrossValidationDataset):

    def __init__(self, **kwargs):
        kwargs['root'] = './dataset/processed/samples'
        kwargs['loader'] = loader
        kwargs['transform'] = transform()
        super().__init__(**kwargs)

    def __getitem__(self, index):
        filepath, label = self.samples[index]
        image, mask = self.loader(filepath)
        image, mask = self.transform(image, mask)
        return image, mask, label
