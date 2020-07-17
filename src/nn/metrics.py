import torch
import numpy as np
from catalyst.contrib.utils.confusion_matrix import calculate_confusion_matrix_from_tensors, calculate_tp_fp_fn
eps = 1e-7

class DiceLoss(torch.nn.Module):

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), 'Size mismatch'
        # y_pred = y_pred[:, 0].contiguous().view(-1)
        # y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice

class IoULoss(torch.nn.Module):

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), 'Size mismatch'
        # y_pred = y_pred[:, 0].contiguous().view(-1)
        # y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        iou = (intersection + eps) / (union - intersection + eps)
        return 1 - iou

detach = lambda t: t.detach().cpu()
threshold = lambda t: (t > 0.5).long()
to_mask = lambda t: threshold(detach(t))
softmax = lambda t: torch.softmax(detach(t))

# def to_class(t):
#     probs = softmax(t)
#     return (probs == probs.max()).long()

def dice_and_iou(outputs, targets):
    outputs, targets = map(to_mask, (outputs, targets))
    intersection = (outputs & targets).float().sum()
    union = (outputs | targets).float().sum()
    dice = (2 * intersection + eps) / (union + intersection + eps)
    iou = (intersection + eps) / (union + eps)
    return dice, iou

def score(outputs, targets, predicitons, labels):
    outputs, targets = map(lambda t: to_mask(t).numpy(), (outputs, targets))
    intersection = np.count_nonzero(np.logical_and(targets, outputs))
    union = np.count_nonzero(np.logical_or(targets, outputs))
    mean = np.mean((intersection + eps) / (union + eps))

    predicitons, labels = map(softmax, (predicitons, labels))
    confusion_matrix = calculate_confusion_matrix_from_tensors(predicitons, labels)
    tp = calculate_tp_fp_fn(confusion_matrix)['true_positives']
    print(tp)

    return mean
    # predicitons, labels = map(lambda t: to_class(t).numpy(), (predicitons, labels))
    # confusion_matrix = 

# def iou(outputs, targets):
#     outputs, targets = map(threshold, (outputs, targets))
#     intersection = (outputs & targets).float().sum()
#     union = (outputs | targets).float().sum()
#     iou = (intersection + eps) / (union - intersection + eps)
#     return iou

# def dice(outputs, targets):
#     outputs, targets = map(threshold, (outputs, targets))
#     intersection = (outputs & targets).float().sum()
#     union = (outputs | targets).float().sum()
#     dice = (2 * intersection + eps) / (union + intersection + eps)
#     return dice
