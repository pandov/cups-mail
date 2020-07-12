# from catalyst.contrib.nn import IoULoss, DiceLoss
import torch
import torchvision
from torch.nn import CrossEntropyLoss
from .loss import DiceLoss
from .dataset import BACTERIA

def get_segmentation_components(n):
    if n == 1:
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        return model, optimizer, scheduler

def get_classification_model(n, num_classes=6):
    if n == 1:
        model = torchvision.models.resnet50(pretrained=True)
        model.train()
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
        return model, optimizer, scheduler

def get_class_names():
    return [
        'c_kefir',
        'ent_cloacae',
        'klebsiella_pneumoniae',
        'moraxella_catarrhalis',
        'staphylococcus_aureus',
        'staphylococcus_epidermidis',
    ]
