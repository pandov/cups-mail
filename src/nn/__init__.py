# from catalyst.contrib.nn import IoULoss, DiceLoss
import torch
from torch.nn import CrossEntropyLoss
from .loss import DiceLoss
from .dataset import BACTERIA

def get_segmentation_model():
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)

def get_classification_model(num_classes):
    import torchvision
    model = torchvision.models.resnet50(pretrained=True)
    model.requires_grad_(False)
    num_features = model.fc.in_features
    # model.fc = torch.nn.Sequential(
    #     torch.nn.Linear(num_features, num_features),
    #     torch.nn.Linear(num_features, num_classes),
    # )
    model.fc = torch.nn.Linear(num_features, num_classes)
    return model

def get_class_names():
    return [
        'c_kefir',
        'ent_cloacae',
        'klebsiella_pneumoniae',
        'moraxella_catarrhalis',
        'staphylococcus_aureus',
        'staphylococcus_epidermidis',
    ]
