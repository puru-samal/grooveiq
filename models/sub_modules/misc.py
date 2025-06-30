import torch.nn as nn

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