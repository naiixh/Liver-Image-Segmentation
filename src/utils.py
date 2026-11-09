"""
=========================================================
GLOBAL UTILITY FUNCTIONS
Includes: Functions for calculating metrics (IoU, Dice),
visualization, and any other helper functions used across the project.
=========================================================

"""

"""---- EVALUATION METRICS ----"""
import torch
import torch.nn.functional as F

def pixel_accuracy(pred, target):
    pred_class = torch.argmax(pred, dim=1)
    correct = (pred_class == target).float()
    return correct.sum() / correct.numel()

def iou_score(pred, target, eps=1e-6):
    pred_class = torch.argmax(pred, dim=1)
    
    intersection = ((pred_class == 1) & (target == 1)).sum().float()
    union = ((pred_class == 1) | (target == 1)).sum().float()
    return (intersection + eps) / (union + eps)


"""---- EVALUATION ----"""

def evaluate(model, val_loader, device, saveResults=False, results=[]):
    model.eval()
    total_iou = 0.0
    total_pa = 0.0
    count = 0
    
    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs = imgs.to(device)
            masks = masks.to(device)
    
            logits = model(imgs)
            if logits.shape[2:] != masks.shape[1:]:
                logits = F.interpolate(logits, size=masks.shape[1:], mode='bilinear', align_corners=False)
            preds = torch.argmax(logits, dim=1) 

            iou = iou_score(logits, masks).item()
            pa = pixel_accuracy(logits, masks).item()
            total_iou += iou
            total_pa += pa
            count += 1

            if saveResults:
                results.append({
                    "img": imgs[0].cpu(),
                    "mask": masks[0].cpu(),
                    "pred": preds[0].cpu(),
                    "iou": iou,
                    "pa": pa
                })
    
    mIoU = total_iou / count
    mPA = total_pa / count

    return mIoU, mPA

