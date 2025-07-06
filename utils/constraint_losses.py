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

    def l1_temporal_diff(self, z, dim=-1):
        """
        L1 temporal difference loss.
        Penalizes absolute differences between consecutive timesteps per latent dimension.
        Encourages sparse changes over time (i.e., piecewise constant behavior).
        Args:
            z: Tensor of shape (B, T, num_button, M)
            dim: Dimension over which to compute the loss
        Returns:
            Loss tensor of shape (B)
        """
        diff = z[:, 1:] - z[:, :-1]                         
        loss_per_step = torch.sum(torch.abs(diff), dim=dim)  
        return self._reduce(loss_per_step)

    def l2_temporal_diff(self, z, dim=-1):
        """
        L2 temporal difference loss.
        Penalizes squared differences between consecutive timesteps per latent dimension.
        Encourages smooth, gradual changes over time.
        Args:
            z: Tensor of shape (B, T, num_button, M)
            dim: Dimension over which to compute the loss
        Returns:
            Loss tensor of shape (B)
        """
        diff = z[:, 1:] - z[:, :-1]
        loss_per_step = torch.sum(diff ** 2, dim=dim)
        return self._reduce(loss_per_step)

    def group_sparse_temporal_diff(self, z, dim=-1):
        """
        Group sparse temporal difference loss.
        Penalizes L2 norm of full vector difference between consecutive timesteps.
        Encourages piecewise constant segments globally across latent vector z.
        Args:
            z: Tensor of shape (B, T, num_button, M)
            dim: Dimension over which to compute the loss
        Returns:
            Loss tensor of shape (B)
        """
        diff = z[:, 1:] - z[:, :-1]
        loss_per_step = torch.norm(diff, p=2, dim=dim) 
        return self._reduce(loss_per_step)

    def l1_sparsity_time(self, z, dim=-1):
        """
        L1 sparsity penalty on temporal dimension.
        Encourages sparsity in activation per timestep (many dimensions close to zero).
        Useful to make z sparse across features at each time.
        Args:
            z: Tensor of shape (B, T, num_button, M)
            dim: Dimension over which to compute the loss
        Returns:
            Loss tensor of shape (B)
        """
        return self._reduce(torch.norm(z, p=1, dim=dim)) 

    def l2_sparsity_time(self, z, dim=-1):
        """
        L2 sparsity penalty on temporal dimension.
        Encourages smaller magnitudes per timestep (shrinks all features together).
        Useful to reduce overall activation energy but without strict sparsity.
        Args:
            z: Tensor of shape (B, T, num_button, M)
            dim: Dimension over which to compute the loss
        Returns:
            Loss tensor of shape (B)
        """
        return self._reduce(torch.norm(z, p=2, dim=dim)) 
