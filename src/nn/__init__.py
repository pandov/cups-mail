# from catalyst.contrib.nn import IoULoss, DiceLoss
import torch
torch.cuda.empty_cache()
from .dataset import BACTERIA
from catalyst.dl import SupervisedRunner, ConfusionMatrixCallback, IouCallback
from catalyst.utils import set_global_seed, prepare_cudnn
prepare_cudnn(deterministic=True)
set_global_seed(7)

class_names = [
    'c_kefir',
    'ent_cloacae',
    'klebsiella_pneumoniae',
    'moraxella_catarrhalis',
    'staphylococcus_aureus',
    'staphylococcus_epidermidis',
]

def get_classification_model(n, num_classes=6):
    from torchvision import models
    if n == 1:
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif n == 2:
        model = models.vgg11(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif n == 3:
        model = models.vgg13(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif n == 4:
        model = models.vgg16(pretrained=True)
        model.features.requires_grad_(False)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif n == 5:
        model = models.alexnet(pretrained=True)
        model.features.requires_grad_(False)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def get_segmentation_model(n):
    if n == 1:
        return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)

def get_optimizer(n, model):
    if n == 1:
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    elif n == 2:
        return torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.99)
    elif n == 3:
        return torch.optim.RMSprop(model.parameters(), lr=1e-3)
    else:
        return None

def get_scheduler(n, optimizer):
    if n == 1:
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
    elif n == 2:
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=0.00001, verbose=True)
    else:
        return None

def get_segmentation_components(m, o=-1, s=-1):
    from .loss import DiceLoss
    model = get_segmentation_model(m)
    optimizer = get_optimizer(o, model)
    scheduler = get_scheduler(s, optimizer)
    criterion = DiceLoss()
    callbacks = [IouCallback()]
    return model, optimizer, scheduler, criterion, callbacks

def get_classification_components(m, o=-1, s=-1):
    from torch.nn import CrossEntropyLoss
    model = get_classification_model(m)
    criterion = CrossEntropyLoss()
    optimizer = get_optimizer(o, model)
    scheduler = get_scheduler(s, optimizer)
    callbacks = [ConfusionMatrixCallback(class_names=class_names)]
    return model, optimizer, scheduler, criterion, callbacks
