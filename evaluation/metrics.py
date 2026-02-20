import torch
import torch.nn as nn

def compute_metrics(preds, targets, threshold=0.5):
    """
    Compute binary classification metrics for segmentation.
    preds: Raw logits or probabilities (B, 1, H, W)
    targets: Ground truth binary mask (B, 1, H, W)
    """
    with torch.no_grad():
        # Ensure preds are probabilities
        if preds.max() > 1.0 or preds.min() < 0.0:
            preds = torch.sigmoid(preds)
            
        # Binarize predictions
        preds_bin = (preds >= threshold).float()
        targets = targets.float()
        
        # True Positives, False Positives, False Negatives, True Negatives
        tp = torch.sum(preds_bin * targets)
        fp = torch.sum(preds_bin * (1 - targets))
        fn = torch.sum((1 - preds_bin) * targets)
        tn = torch.sum((1 - preds_bin) * (1 - targets))
        
        # Add epsilon to prevent division by zero
        eps = 1e-7
        
        # Dice Coefficient = 2*TP / (2*TP + FP + FN)
        dice = (2 * tp + eps) / (2 * tp + fp + fn + eps)
        
        # IoU (Jaccard Index) = TP / (TP + FP + FN)
        iou = (tp + eps) / (tp + fp + fn + eps)
        
        # Precision = TP / (TP + FP)
        precision = (tp + eps) / (tp + fp + eps)
        
        # Recall (Sensitivity) = TP / (TP + FN)
        recall = (tp + eps) / (tp + fn + eps)
        
        # Specificity = TN / (TN + FP)
        specificity = (tn + eps) / (tn + fp + eps)
        
        return {
            'dice': dice.item(),
            'iou': iou.item(),
            'precision': precision.item(),
            'recall': recall.item(),
            'specificity': specificity.item()
        }

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        # Convert logits to probabilities
        probs = torch.sigmoid(logits)
        
        # Flatten label and prediction tensors
        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)
        
        intersection = (probs_flat * targets_flat).sum()
        
        # 2 * (P * T) / (P + T)
        dice_score = (2. * intersection + self.smooth) / (probs_flat.sum() + targets_flat.sum() + self.smooth)
        
        # Loss is 1 - Dice
        return 1. - dice_score

class CombinedLoss(nn.Module):
    """
    Combined BCE with Logits and Dice Loss.
    Addresses class imbalance by balancing pixel-wise cross-entropy with overlap tracking.
    """
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super(CombinedLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.dice_loss = DiceLoss()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        
    def forward(self, logits, targets):
        bce = self.bce_loss(logits, targets)
        dice = self.dice_loss(logits, targets)
        return self.bce_weight * bce + self.dice_weight * dice
