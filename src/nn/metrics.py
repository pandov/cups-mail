import torch
import numpy as np

smooth = 1e-7
prepare = lambda t: t.detach().squeeze(1)

def threshold(tensor, t=0.4):
    # tensor.requires_grad = False
    tensor[tensor >= t] = 1
    tensor[tensor < t] = 0
    return tensor.long()

def iou(outputs, targets):
    outputs, targets = prepare(outputs), prepare(targets)
    outputs, targets = threshold(outputs), threshold(targets)
    intersection = (outputs & targets).float().sum()
    union = (outputs | targets).float().sum()
    iou = (intersection + smooth) / (union + smooth)
    return iou
