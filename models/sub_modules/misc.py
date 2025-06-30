import torch.nn as nn
import torch

class SpatioTemporalMotifEncoder(nn.Module):
    """
    ### Inputs:
        x: Tensor of shape (B, T, E, M)
            - B: batch size
            - T: number of time steps
            - E: number of drum instruments
            - M: number of expressive features per instrument
              (e.g., hit presence, velocity, timing offset)

    ### Outputs:
        Tensor of shape (B, T, embed_dim)
            - Per-timestep learned embedding that captures spatial and temporal motif structure
    """

    def __init__(self, E, M, spatial_channels, temporal_dim, embed_dim, kernel_size):
        """
        Args:
            E (int): Number of drum instruments (spatial height).
            M (int): Number of expressive features per instrument (spatial width).
            spatial_channels (int): Number of intermediate spatial feature maps.
            temporal_dim (int): Dimensionality after spatial projection (temporal modeling input).
            embed_dim (int): Output embedding dimension per time step.
            kernel_size (int): Size of temporal convolution window (must be odd for padding symmetry).
        """
        super().__init__()
        self.E = E
        self.M = M

        # Spatial Encoder
        # Applies 2D convolution over (E, M) grid per time step
        self.spatial_encoder = nn.Sequential(
            nn.Conv2d(1, spatial_channels, kernel_size=3, padding=1),   # (B*T, spatial_channels, E, M)
            nn.ReLU(),
            nn.Conv2d(spatial_channels, temporal_dim, kernel_size=1),   # Project to temporal_dim features
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1))  # Reduce (E, M) → (1, 1), outputs (B*T, temporal_dim, 1, 1)
        )

        # Temporal Model
        # Applies 1D convolution over time using learned local filters
        self.temporal_model = nn.Sequential(
            nn.Conv1d(
                in_channels=temporal_dim,
                out_channels=temporal_dim,
                kernel_size=kernel_size,
                padding=kernel_size // 2  # Preserve sequence length
            ),
            nn.ReLU(),
            nn.Conv1d(
                in_channels=temporal_dim,
                out_channels=embed_dim,
                kernel_size=1
            )  # Final projection to per-timestep embedding
        )

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)

        Returns:
            Tensor of shape (B, T, embed_dim)
        """
        B, T, E, M = x.shape
        assert E == self.E and M == self.M, "Input shape mismatch."

        # Step 1: Spatial Encoding per frame
        # Flatten batch and time to treat each frame as an image: (B*T, 1, E, M)
        x = x.view(B * T, 1, E, M)
        x = self.spatial_encoder(x)  # (B*T, temporal_dim, 1, 1)
        x = x.view(B, T, -1)         # (B, T, temporal_dim)

        # Step 2: Temporal Modeling
        x = x.permute(0, 2, 1)       # (B, temporal_dim, T) for Conv1D
        x = self.temporal_model(x)   # (B, embed_dim, T)
        x = x.permute(0, 2, 1)       # (B, T, embed_dim)

        return x
    

class IntegerQuantizer(nn.Module):
    """
    A quantization module that maps continuous values in [-1, 1] to discrete bins,
    and vice versa. This is typically used to learn discrete control signals
    (e.g., buttons) while training with gradient-based methods.

    It uses the straight-through estimator (Bengio et al., 2013) during backpropagation,
    so gradients are passed through the quantization step as if it were the identity function.

    Args:
        num_bins (int): The number of discrete bins to quantize into.
                        Should match the number of discrete control tokens (e.g., 8 buttons).
    """
    def __init__(self, num_bins):
        super().__init__()
        self.num_bins = num_bins

    def real_to_discrete(self, x, eps=1e-6):
        """
        Maps continuous input x ∈ [-1, 1] to integer indices in [0, num_bins - 1].

        Steps:
        - Normalize x from [-1, 1] to [0, 1]
        - Scale to [0, num_bins - 1]
        - Round to nearest integer index
        """
        x = (x + 1) / 2                      # Map from [-1, 1] → [0, 1]
        x = torch.clamp(x, 0, 1)             # Ensure input is bounded
        x *= self.num_bins - 1               # Scale to [0, num_bins - 1]
        x = (torch.round(x) + eps).long()    # Round and convert to int
        return x

    def discrete_to_real(self, x):
        """
        Maps integer bin indices in [0, num_bins - 1] back to continuous values in [-1, 1].

        This is the inverse of `real_to_discrete` using uniform bin spacing.
        """
        x = x.float()
        x /= self.num_bins - 1               # Map to [0, 1]
        x = (x * 2) - 1                      # Map to [-1, 1]
        return x

    def forward(self, x):
        """
        Quantizes continuous input `x ∈ [-1, 1]` into discrete bins,
        but uses the straight-through estimator to pass gradients.

        In the forward pass:
        - Round input to nearest discrete bin
        - Convert back to continuous value (quantized)
        - Compute the difference between quantized and original value
        - Add that difference back to original input (no-op in forward, identity in backward)

        This allows the model to be trained as if quantization is differentiable.

        Returns:
            Tensor of same shape as input, with values clamped to discrete levels,
            but gradients flowing as if through identity.
        """
        with torch.no_grad():
            x_disc = self.real_to_discrete(x)        # Discrete integer indices
            x_quant = self.discrete_to_real(x_disc)  # Quantized values in [-1, 1]
            x_quant_delta = x_quant - x              # Forward difference (stopping gradient here)

        # Straight-through estimator: forward uses quantized, backward sees identity
        x = x + x_quant_delta

        return x