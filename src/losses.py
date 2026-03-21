import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    """
    alpha controls False Positives penalty
    beta controls False Negatives penalty
    for tumors: beta > alpha (missing tumor is worse than false alarm)
    """

    def __init__(self, alpha=0.3, beta=0.7, smooth=1.0):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)

        # Flatten spatial dims per batch element
        dims = (1, 2, 3)
        TP = (probs * targets).sum(dims)
        FP = (probs * (1 - targets)).sum(dims)
        FN = ((1- probs) * targets).sum(dims)

        tversky = (TP + self.smooth) / (
            TP + self.alpha * FP + self.beta * FN + self.smooth
        )

        return (1-tversky).mean()
    
class FocalLoss(nn.Module):
    """
    Focuses training on hard/misclassified examples.
    gamma=2 is standard, higher = more focus on hard examples.
    """

    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma 

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, reduction='none'
        )

        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        loss = focal_weight * bce
        return loss.mean()
    
class TverskyFocalLoss(nn.Module):
    """
    Combined loss = Tversky + Focal
    - Tversky: handles class imbalance, penalizes missed tumors
    - Focal: handles hard examples, small tumor boundaries

    tversky_weight = 0.7 means Tversky dominates (recommended)
    """

    def __init__(
            self, 
            tversky_alpha=0.3, # FP panalty (low=less penalty for false alarms)
            tversky_beta=0.7, # FN penalty (high = heavy penalty for missing tumor)
            focal_alpha=0.25,
            focal_gamma=2.0,
            tversky_weight=0.7, # How much Twersky contributes vs Focal
            smooth=1.0
    ): 
        super().__init__()
        self.tversky = TverskyLoss(tversky_alpha, tversky_beta, smooth)
        self.focal = FocalLoss(focal_alpha, focal_gamma)
        self.tversky_weight = tversky_weight

    def forward(self, logits, targets):
        t_loss = self.tversky(logits, targets)
        f_loss = self.focal(logits, targets)

        loss = (self.tversky_weight * t_loss + (1 - self.tversky_weight) * f_loss)
        return loss
    
def dice_score(logits, targets, thresold = 0.5):
    """
    Fixed version of your dice_score:
    - Handles empty masks properly
    - Per-patient averaging not per-slice
    """

    probs = torch.sigmoid(logits)
    preds = (probs > thresold).float()

    dims = (1, 2, 3)

    intersection = (preds * targets).sum(dims)
    union = preds.sum(dims) + targets.sum(dims)

    # If both pred and target are empty → perfect score (1.0)
    # If only one is empty → score is 0.0 (bad prediction)
    dice = torch.zeros_like(intersection)

    both_empty = (union == 0)
    has_content = ~both_empty

    dice[both_empty] = 1.0
    dice[has_content] = (
        (2 * intersection[has_content] + 1e-5) /
        (union[has_content] + 1e-5)
    )

    return dice.mean().item()
