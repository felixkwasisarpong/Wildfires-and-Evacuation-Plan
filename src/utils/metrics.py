from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import torch
def precision(y_true, y_pred):
    return precision_score(y_true, y_pred)

def recall(y_true, y_pred):
    return recall_score(y_true, y_pred)

def f1(y_true, y_pred):
    return f1_score(y_true, y_pred)

def pr_auc(y_true, y_pred):
    return roc_auc_score(y_true, y_pred)


def dice_coefficient(pred, target, smooth=1e-6):
    # Flatten the tensors to treat them as one-dimensional arrays
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = torch.sum(pred * target)
    dice = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    
    return dice


def intersection_over_union(pred, target, smooth=1e-6):
    # Flatten the tensors to treat them as one-dimensional arrays
    pred = pred.view(-1)
    target = target.view(-1)
    
    intersection = torch.sum(pred * target)
    union = torch.sum(pred) + torch.sum(target) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return iou
