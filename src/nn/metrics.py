import torch

smooth = 1e-7

class DiceLoss(torch.nn.Module):

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), 'Size mismatch'
        # y_pred = y_pred[:, 0].contiguous().view(-1)
        # y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice

class IoULoss(torch.nn.Module):

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), 'Size mismatch'
        # y_pred = y_pred[:, 0].contiguous().view(-1)
        # y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        iou = (intersection + smooth) / (union - intersection + smooth)
        return 1 - iou

def threshold(tensor):
    tensor = tensor.detach().cpu()#.squeeze(1)
    # tensor.requires_grad = False
    return (tensor > 0.5).long()

def dice_and_iou(outputs, targets):
    outputs, targets = map(threshold, (outputs, targets))
    intersection = (outputs & targets).float().sum()
    union = (outputs | targets).float().sum()
    dice = (2 * intersection + smooth) / (union + intersection + smooth)
    iou = (intersection + smooth) / (union - intersection + smooth)
    return dice, iou

# def iou(outputs, targets):
#     outputs, targets = map(threshold, (outputs, targets))
#     intersection = (outputs & targets).float().sum()
#     union = (outputs | targets).float().sum()
#     iou = (intersection + smooth) / (union - intersection + smooth)
#     return iou

# def dice(outputs, targets):
#     outputs, targets = map(threshold, (outputs, targets))
#     intersection = (outputs & targets).float().sum()
#     union = (outputs | targets).float().sum()
#     dice = (2 * intersection + smooth) / (union + intersection + smooth)
#     return dice
