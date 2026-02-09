import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits, targets):

        # convert logits to probabilities
        probs = torch.sigmoid(logits)

        # Flatten 
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )

        return 1 - dice
    
def dice_score(logits, targets, thresold=0.5):
    probs = torch.sigmoid(logits)
    preds = (probs > thresold).float()

    preds = preds.view(-1)
    targets = targets.view(-1)

    intersection = (preds * targets).sum()
    dice = (2.0 * intersection)/(preds.sum() + targets.sum() + 1e-6)

    return dice.item()
