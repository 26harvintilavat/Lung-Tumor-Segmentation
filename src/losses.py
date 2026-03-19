import torch
import torch.nn as nn
import torch.nn.functional as F


class SoftDiceLoss(nn.Module):
    def __init__(self, smooth=1e-4):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        probs = torch.sigmoid(logits)

        # compute dice per batch element
        dims = (1, 2, 3)

        intersection = (probs * targets).sum(dims)
        union = probs.sum(dims) + targets.sum(dims)

        dice = (2 * intersection + self.smooth) / (union + self.smooth)

        loss = 1 - dice
        return loss.mean()


class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([3.0]))
        self.dice = SoftDiceLoss()
        self.bce_weight = bce_weight

    def forward(self, logits, targets):

        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)

        loss = self.bce_weight * bce_loss + (1 - self.bce_weight) * dice_loss

        return loss


def dice_score(logits, targets, threshold=0.5):

    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()

    dims = (1, 2, 3)

    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)

    dice = (2 * intersection + 1e-5) / (union + 1e-5)
    dice[union == 0] = 1

    return dice.mean().item()