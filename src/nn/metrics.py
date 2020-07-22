import torch
import numpy as np
from catalyst.contrib.utils.confusion_matrix import calculate_confusion_matrix_from_tensors, calculate_tp_fp_fn
eps = 1e-7

class DiceLoss(torch.nn.Module):

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), 'Size mismatch'
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dice = (2 * intersection + eps) / (union + eps)
        return 1 - dice

class IoULoss(torch.nn.Module):

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), 'Size mismatch'
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        iou = (intersection + eps) / (union - intersection + eps)
        return 1 - iou

detach = lambda t: t.detach().cpu()
threshold = lambda t, k: (t > k).long()
to_mask = lambda t, k: threshold(detach(t), k)

def score_clf(predicitons, labels):
    predicitons, labels = map(detach, (predicitons, labels))
    # probabilities = torch.softmax(predicitons, dim=0)
    confusion_matrix = calculate_confusion_matrix_from_tensors(predicitons, labels)
    p = calculate_tp_fp_fn(confusion_matrix)
    tp, fp, fn = p['true_positives'], p['false_positives'], p['false_negatives']
    precision = (tp + eps) / (tp + fp + eps)
    return precision

def score_aux(outputs, targets, k=0.5):
    outputs, targets = map(lambda t: to_mask(t, k).numpy(), (outputs, targets))
    intersection = np.count_nonzero(np.logical_and(targets, outputs))
    union = np.count_nonzero(np.logical_or(targets, outputs))
    iou = (intersection + eps) / (union + eps)
    # mean = np.mean(iou)
    return iou

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
