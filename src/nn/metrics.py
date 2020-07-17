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

# def dice_and_iou(outputs, targets):
#     outputs, targets = map(to_mask, (outputs, targets))
#     intersection = (outputs & targets).float().sum()
#     union = (outputs | targets).float().sum()
#     dice = (2 * intersection + eps) / (union + intersection + eps)
#     iou = (intersection + eps) / (union + eps)
#     return dice, iou

def score_clf(predicitons, labels):
    predicitons, labels = map(detach, (predicitons, labels))
    probabilities = torch.softmax(predicitons, dim=0)
    confusion_matrix = calculate_confusion_matrix_from_tensors(probabilities, labels)
    p = calculate_tp_fp_fn(confusion_matrix)
    tp, fp, fn = p['true_positives'], p['false_positives'], p['false_negatives']
    precision = tp / (tp + fp + eps)
    return precision

def score_aux(outputs, targets):
    outputs, targets = map(lambda t: to_mask(t).numpy(), (outputs, targets))
    intersection = np.count_nonzero(np.logical_and(targets, outputs))
    union = np.count_nonzero(np.logical_or(targets, outputs))
    iou = intersection / (union + eps)
    mean = np.mean(iou)
    return mean

# def score_global(outputs, targets, predicitons, labels):
#     return score_aux(outputs, targets) + score_clf(predicitons, labels)

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
