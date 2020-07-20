# from catalyst.contrib.nn import IoULoss, DiceLoss
import torch
torch.cuda.empty_cache()
import segmentation_models_pytorch as segmentation
from catalyst.dl import Runner, SupervisedRunner
from catalyst.utils import set_global_seed, prepare_cudnn
from .dataset import BACTERIA
from .metrics import score_aux, score_clf

prepare_cudnn(deterministic=True)
set_global_seed(3)

def get_class_names():
    return [
        'c_kefir',
        'ent_cloacae',
        'klebsiella_pneumoniae',
        'moraxella_catarrhalis',
        'staphylococcus_aureus',
        'staphylococcus_epidermidis',
    ]

def get_classification_model(name, num_classes=6):
    from torchvision import models
    if name == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif name == 'vgg1':
        model = models.vgg11(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == 'vgg13':
        model = models.vgg13(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif name == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.features.requires_grad_(False)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    return model

def get_segmentation_model(name, encoder='resnet34'):
    if name == 'brain':
        return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
    elif name == 'unet':
        return segmentation.Unet(encoder, activation='sigmoid', in_channels=1)
    elif name == 'fpn':
        return segmentation.FPN(encoder, activation='sigmoid', in_channels=1)

def get_multimodel(name, encoder):
    kwargs = dict(activation='sigmoid', in_channels=1, classes=1, aux_params=dict(classes=6))
    if name == 'unet':
        return segmentation.Unet(encoder, **kwargs)
    elif name == 'fpn':
        return segmentation.FPN(encoder, **kwargs)
    elif name == 'linknet':
        return segmentation.Linknet(encoder, **kwargs)
    elif name == 'pspnet':
        return segmentation.PSPNet(encoder, **kwargs)

def get_optimizer(name, model):
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=1e-3)
    elif name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=1e-3)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=1e-3)
    else:
        return None

def get_scheduler(name, optimizer):
    if name == 'steplr':
        # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)
    elif name == 'reducelronplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-5, verbose=True)
    else:
        return None

def get_loaders(keys, batch_size):
    train_dataset = BACTERIA('train', keys)
    valid_dataset = BACTERIA('valid', keys)
    return {
        'train': train_dataset.data_loader(batch_size=batch_size, shuffle=True, drop_last=True),
        'valid': valid_dataset.data_loader(),
    }

def get_dict_components(o, s, model, criterion, callbacks):
    optimizer = get_optimizer(o, model)
    scheduler = get_scheduler(s, optimizer)
    return {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'criterion': criterion,
        'callbacks': callbacks,
    }

def get_segmentation_components(m, e, o=None, s=None):
    from catalyst.dl import IouCallback
    from .metrics import DiceLoss
    return get_dict_components(o, s,
        get_segmentation_model(m, e), DiceLoss(), callbacks=None)

def get_classification_components(m, o=None, s=None):
    from catalyst.dl import ConfusionMatrixCallback
    from torch.nn import CrossEntropyLoss
    return get_dict_components(o, s,
        get_classification_model(m), CrossEntropyLoss(), [ConfusionMatrixCallback(class_names=get_class_names())])

def get_multimodel_components(m, e, o=None, s=None):
    from catalyst.dl import ConfusionMatrixCallback
    from torch.nn import CrossEntropyLoss
    from .metrics import DiceLoss, IoULoss
    criterion = {
        'iou': IoULoss(),
        'dice': DiceLoss(),
        'crossentropy': CrossEntropyLoss(),
    }
    return get_dict_components(o, s,
        get_multimodel(m, e), criterion, [ConfusionMatrixCallback(class_names=get_class_names())])
