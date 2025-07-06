import torch
import torch.nn as nn

class ConstraintLosses(nn.Module):
    """
    Generic class for different custom losses, including temporal difference losses.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", "sum"], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def _reduce(self, loss_per_step):
        if self.reduction == "mean":
            return loss_per_step.mean()
        else:
            return loss_per_step.sum()

    def l2_temporal_diff(self, z):
        """
        Penalize large differences over time using L2 norm squared.
        """
        diff = z[:, 1:] - z[:, :-1]
        loss_per_step = torch.sum(diff ** 2, dim=-1)
        return self._reduce(loss_per_step)

    def l1_temporal_diff(self, z):
        """
        Penalize differences over time using L1 norm (promotes sparsity in changes).
        """
        diff = z[:, 1:] - z[:, :-1]
        loss_per_step = torch.sum(torch.abs(diff), dim=-1)
        return self._reduce(loss_per_step)

    def group_sparse_temporal_diff(self, z):
        """
        Penalize group differences (L2 norm on vector differences), encouraging piecewise constant behavior.
        """
        diff = z[:, 1:] - z[:, :-1]
        loss_per_step = torch.norm(diff, p=2, dim=-1)
        return self._reduce(loss_per_step)

    def l1_sparsity(self, z):
        """
        L1 sparsity penalty directly on z to encourage sparsity in activations.
        """
        return self._reduce(torch.abs(z))

    def l1_sparsity_time(self, z):
        """
        L1 sparsity penalty on temporal dimension of z to encourage sparsity in activations.
        """
        return self._reduce(torch.norm(z, p=1, dim=-1))
    
    def l2_sparsity_time(self, z):
        """
        L2 sparsity penalty directly on z to encourage sparsity in activations.
        """
        return self._reduce(torch.norm(z, p=2, dim=-1))
    
    def margin_loss(self, z):
        """
        Margin loss to constrain the latent to be between -1 and 1.
        """
        return self._reduce(torch.square(torch.maximum(torch.abs(z) - 1, torch.zeros_like(z))))
