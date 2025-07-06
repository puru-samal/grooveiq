import torch
import torch.nn as nn

class ConstraintLosses(nn.Module):
    """
    Generic class for different custom constraint or regularization losses.
    """

    def __init__(self, reduction="mean"):
        super().__init__()
        assert reduction in ["mean", "sum"], "reduction must be 'mean' or 'sum'"
        self.reduction = reduction

    def _reduce(self, loss_per_step):
        """
        Reduce loss across batch and time, either mean or sum.
        """
        if self.reduction == "mean":
            return loss_per_step.mean()
        else:
            return loss_per_step.sum()

    def l1_temporal_diff(self, z):
        """
        L1 temporal difference loss.
        Penalizes absolute differences between consecutive timesteps per latent dimension.
        Encourages sparse changes over time (i.e., piecewise constant behavior).
        """
        diff = z[:, 1:] - z[:, :-1]                         # (B, T-1, D)
        loss_per_step = torch.sum(torch.abs(diff), dim=-1)  # sum over D -> (B, T-1)
        return self._reduce(loss_per_step)

    def l2_temporal_diff(self, z):
        """
        L2 temporal difference loss.
        Penalizes squared differences between consecutive timesteps per latent dimension.
        Encourages smooth, gradual changes over time.
        """
        diff = z[:, 1:] - z[:, :-1]
        loss_per_step = torch.sum(diff ** 2, dim=-1)
        return self._reduce(loss_per_step)

    def group_sparse_temporal_diff(self, z):
        """
        Group sparse temporal difference loss.
        Penalizes L2 norm of full vector difference between consecutive timesteps.
        Encourages piecewise constant segments globally across latent vector z.
        """
        diff = z[:, 1:] - z[:, :-1]
        loss_per_step = torch.norm(diff, p=2, dim=-1)  # (B, T-1)
        return self._reduce(loss_per_step)

    def l1_sparsity_time(self, z):
        """
        L1 sparsity penalty on temporal dimension.
        Encourages sparsity in activation per timestep (many dimensions close to zero).
        Useful to make z sparse across features at each time.
        """
        return self._reduce(torch.norm(z, p=1, dim=-1))  # (B, T)

    def l2_sparsity_time(self, z):
        """
        L2 sparsity penalty on temporal dimension.
        Encourages smaller magnitudes per timestep (shrinks all features together).
        Useful to reduce overall activation energy but without strict sparsity.
        """
        return self._reduce(torch.norm(z, p=2, dim=-1))  # (B, T)
