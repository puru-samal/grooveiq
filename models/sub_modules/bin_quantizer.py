import torch
import torch.nn as nn


class LearnableBinsQuantizer(nn.Module):
    """
    A learnable scalar quantizer module using bins with straight-through estimator (STE).

    - Maps continuous input values into discrete bins.
    - Instead of fixed bin centers, the representative value of each bin is learnable.
    - Uses a straight-through estimator so gradients can flow through the discrete binning step.
    """

    def __init__(self, num_bins, min_val=0.0, max_val=1.0):
        """
        Args:
            num_bins (int): Number of bins to quantize into.
            min_val (float): Minimum possible input value (defines lower bound of bins).
            max_val (float): Maximum possible input value (defines upper bound of bins).
        """
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        # Initialize bin representative values linearly spaced between min_val and max_val
        initial_bins = torch.linspace(min_val, max_val, num_bins)
        self.bin_values = nn.Parameter(initial_bins)
        # These bin values will be learned during training to optimally represent data.

    def forward(self, x):
        """
        Args:
            x: Input tensor of arbitrary shape (...), expected in [min_val, max_val].
        Returns:
            Quantized tensor of same shape as x, but using learned bin representative values.
        """

        # -------- Normalize input to [0, 1] range for comparison --------
        x_norm = (x - self.min_val) / (self.max_val - self.min_val)
        x_norm = torch.clamp(x_norm, 0, 1)

        # -------- Create reference bins spaced in [0, 1] --------
        reference_bins = torch.linspace(0, 1, self.num_bins, device=x.device)

        # -------- Compute distances to each bin center --------
        # Shape: (..., num_bins) after unsqueeze
        distances = torch.abs(x_norm.unsqueeze(-1) - reference_bins)

        # -------- Find index of closest bin --------
        bin_idx = torch.argmin(distances, dim=-1)  # shape: (...)

        # -------- Use corresponding learned bin value --------
        bin_value = self.bin_values[bin_idx]  # shape: (...)

        # -------- Straight-through estimator (STE) --------
        # Forward: use bin_value (quantized).
        # Backward: gradients flow as if original x was used.
        out = x + (bin_value - x).detach()

        return out
