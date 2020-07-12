# from catalyst.contrib.nn import IoULoss, DiceLoss
import torch
import torchvision
from torch.nn import CrossEntropyLoss
from .loss import DiceLoss
from .dataset import BACTERIA

def get_scheduler(n, optimizer):
    if n == 1:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

def get_optimizer(n, model):
    if n == 1:
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    elif n == 2:
        return torch.optim.RMSprop(model.parameters(), lr=1e-2)

def get_segmentation_components(n):
    if n == 1:
        model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
        optimizer = get_optimizer(1, model)
        scheduler = get_scheduler(1, optimizer)

    model.train()
    return model, optimizer, scheduler

def get_classification_components(n, num_classes=6):
    if n == 1:
        model = torchvision.models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
        optimizer = get_optimizer(1, model)
        scheduler = get_scheduler(1, optimizer)
    elif n == 2:
        model = torchvision.models.vgg16_bn(pretrained=True)
        optimizer = get_optimizer(1, model)
        scheduler = get_scheduler(1, optimizer)

    model.train()
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
