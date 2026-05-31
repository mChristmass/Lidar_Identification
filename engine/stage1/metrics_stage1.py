import torch

def compute_metrics(logits, labels, valid_mask=None):

    preds = torch.argmax(logits, dim=1)

    preds = preds.view(-1)
    labels = labels.view(-1)

    if valid_mask is not None:

        valid_mask = valid_mask.view(-1)

        preds = preds[valid_mask > 0]
        labels = labels[valid_mask > 0]

    TP = ((preds == 1) & (labels == 1)).sum().item()
    FP = ((preds == 1) & (labels == 0)).sum().item()
    FN = ((preds == 0) & (labels == 1)).sum().item()

    precision = TP / (TP + FP + 1e-6)
    recall = TP / (TP + FN + 1e-6)
    iou = TP / (TP + FP + FN + 1e-6)

    return iou, precision, recall