from torch.nn import Module

class DiceLoss(Module):

    def __init__(self):
        super().__init__()
        self.smooth = 1.0

    def forward(self, y_pred, y_true):
        assert y_pred.size() == y_true.size(), 'Size mismatch'
        y_pred = y_pred[:, 0].contiguous().view(-1)
        y_true = y_true[:, 0].contiguous().view(-1)
        intersection = (y_pred * y_true).sum()
        union = y_pred.sum() + y_true.sum()
        dsc = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1. - dsc
