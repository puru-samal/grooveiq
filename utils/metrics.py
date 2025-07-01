import torch
import torch.nn.functional as F

class DrumMetrics:
    @staticmethod
    def hit_metrics(pred_logits, target_hits, threshold=0.5):
        pred_int = (torch.sigmoid(pred_logits) > threshold).int()
        target_int = target_hits.int()

        tp = ((pred_int == 1) & (target_int == 1)).sum().item()
        fp = ((pred_int == 1) & (target_int == 0)).sum().item()
        fn = ((pred_int == 0) & (target_int == 1)).sum().item()
        tn = ((pred_int == 0) & (target_int == 0)).sum().item()

        acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) > 0 else 0.0
        ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1  = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

        return {"acc": acc, "ppv": ppv, "tpr": tpr, "f1": f1}

    @staticmethod
    def mse_metrics(pred, target):
        return F.mse_loss(pred, target).item()

    @staticmethod
    def mae_metrics(pred, target):
        return F.l1_loss(pred, target).item()

    @staticmethod
    def pearson_corr(pred, target):
        pred_flat = pred.flatten()
        target_flat = target.flatten()
        pred_mean = pred_flat.mean()
        target_mean = target_flat.mean()
        num = ((pred_flat - pred_mean) * (target_flat - target_mean)).sum()
        denom = torch.sqrt(((pred_flat - pred_mean) ** 2).sum() * ((target_flat - target_mean) ** 2).sum())
        return (num / (denom + 1e-8)).item()

    @staticmethod
    def range_diff(pred, target):
        pred_range = (pred.max() - pred.min()).item()
        target_range = (target.max() - target.min()).item()
        return abs(pred_range - target_range)

    @staticmethod
    def percent_within_tolerance(pred, target, tolerance=0.02):
        within_tolerance = (torch.abs(pred - target) < tolerance).float().mean().item()
        return within_tolerance

    @staticmethod
    def ahead_behind_ratio(pred, target):
        ahead = (pred - target < 0).float().mean().item()
        behind = (pred - target > 0).float().mean().item()
        return {"ahead": ahead, "behind": behind}

    @staticmethod
    def perplexity(bce_loss):
        return torch.exp(bce_loss).item()

class RunningAverageMeter:
    def __init__(self):
        self.total = 0.0
        self.count = 0

    def update(self, val, n=1):
        self.total += val * n
        self.count += n

    def average(self):
        return self.total / self.count if self.count > 0 else 0.0