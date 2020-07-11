import torch
from catalyst.contrib.nn import IoULoss#, DiceLoss
from .dataset import BACTERIA
from .loss import DiceLoss

def get_model():
    return torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet', in_channels=3, out_channels=1, init_features=32, pretrained=True)
