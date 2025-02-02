# from catalyst.contrib.nn import IoULoss, DiceLoss
import torch
torch.cuda.empty_cache()
import segmentation_models_pytorch as segmentation
from torch import set_grad_enabled
from catalyst.dl import Runner, SupervisedRunner
from catalyst.utils import get_device, set_global_seed, prepare_cudnn
from .dataset import BACTERIA
from .metrics import score_aux, score_clf

device = get_device()
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
    name = name.lower()
    if 'resnet' in name:
        if '50' in name:
            model = models.resnet50(pretrained=True)
        elif '101' in name:
            model = models.resnet101(pretrained=True)
        model.conv1 = torch.nn.Conv2d(1, model.conv1.out_channels, model.conv1.kernel_size, model.conv1.stride, model.conv1.padding)
        model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
    elif 'vgg' in name:
        if '11' in name:
            model = models.vgg11(pretrained=True)
        elif '13' in name:
            model = models.vgg13(pretrained=True)
        elif '16' in name:
            model = models.vgg16(pretrained=True)
        model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, num_classes)
    elif 'efficientnet' in name:
        from efficientnet_pytorch import EfficientNet
        advprop = 'adv' in name
        if advprop:
            name = name[4:]
        model = EfficientNet.from_pretrained(name, in_channels=1, num_classes=num_classes, advprop=advprop)
    return model

def get_segmentation_model(name, encoder_name=None):
    name = name.lower()
    if name == 'brain':
        return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=1, out_channels=1, init_features=32, pretrained=True)
    else:
        kwargs = dict(encoder_name=encoder_name, activation='sigmoid', in_channels=1)
        if name == 'unet':
            return segmentation.Unet(**kwargs)
        elif name == 'fpn':
            return segmentation.FPN(**kwargs)
        elif name == 'pan':
            return segmentation.PAN(**kwargs)
        elif name == 'linknet':
            return segmentation.Linknet(**kwargs)
        elif name == 'pspnet':
            return segmentation.PSPNet(**kwargs)
        elif name == 'deeplabv3':
            return segmentation.DeepLabV3(**kwargs)

def get_multimodel(name, encoder):
    name = name.lower()
    kwargs = dict(encoder=encoder, activation='sigmoid', in_channels=1, classes=1, aux_params=dict(classes=6))
    if name == 'unet':
        return segmentation.Unet(**kwargs)
    elif name == 'fpn':
        return segmentation.FPN(**kwargs)
    elif name == 'linknet':
        return segmentation.Linknet(**kwargs)
    elif name == 'pspnet':
        return segmentation.PSPNet(**kwargs)

def get_optimizer(name, model):
    if name == 'adam':
        return torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-7)
    elif name == 'sgd':
        return torch.optim.SGD(model.parameters(), lr=1e-3)
    elif name == 'rmsprop':
        return torch.optim.RMSprop(model.parameters(), lr=1e-3)
    else:
        return None

def get_scheduler(name, optimizer):
    if name == 'steplr':
        # return torch.optim.lr_scheduler.StepLR(optimizer, step_size=60, gamma=0.1)
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=70, gamma=0.1)
    elif name == 'reducelronplateau':
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=3, min_lr=1e-5, verbose=True)
    else:
        return None

def get_loaders(batch_size, **kwargs):
    train_dataset = BACTERIA('train', **kwargs)
    valid_dataset = BACTERIA('valid', **kwargs)
    return {
        'train': train_dataset.data_loader(batch_size=batch_size, shuffle=True, drop_last=True),
        'valid': valid_dataset.data_loader(batch_size=min(batch_size, len(valid_dataset))),
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
    from .metrics import DiceLoss, IoULoss
    criterion = {
        'iou': IoULoss(),
        'dice': DiceLoss(),
    }
    return get_dict_components(o, s,
        get_segmentation_model(m, e), criterion, callbacks=None)

def get_classification_components(m, o=None, s=None, weightable=False):
    from catalyst.dl import ConfusionMatrixCallback
    from torch.nn import CrossEntropyLoss
    weight = None
    if weightable:
        weight = torch.tensor([52., 69., 10., 3+29, 1+14., 7+18.], device=device)
        weight /= weight.max()
        weight = 2 - weight
    return get_dict_components(o, s,
        get_classification_model(m), CrossEntropyLoss(weight=weight), [ConfusionMatrixCallback(class_names=get_class_names())])

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
