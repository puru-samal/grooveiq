import torch
import torch.nn as nn
import torch.nn.functional as F
from .axial_attention import *

# -----------------------------------
# 1. Local 2D Convolutional Encoder
# -----------------------------------
class DrumConvEncoder(nn.Module):
    """
    Simple convolutional encoder.
    - Models local spatio-temporal interactions.
    - No global temporal context or long-range instrument mixing.
    """
    def __init__(self, M : int =3, embed_dim : int =128, depth : int =4):
        super().__init__()
        layers = []
        in_ch = M
        for _ in range(depth):
            layers.append(nn.Conv2d(in_ch, embed_dim, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(embed_dim))
            layers.append(nn.LeakyReLU(inplace=True))
            in_ch = embed_dim
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        B, T, E, M = x.shape
        x = x.permute(0, 3, 1, 2)  # B x M x T x E
        x = self.conv(x)
        x = x.permute(0, 2, 3, 1)  # B x T x E x D
        return x


# -----------------------------------
# 2. Temporal-only Transformer Encoder
# -----------------------------------
class DrumTemporalTransformer(nn.Module):
    """
    Transformer encoder across time.
    - Models long-range global temporal dependencies.
    - No explicit modeling of instrument relationships.
    - Instruments are flattened into channels.
    """
    def __init__(self, E=9, M=3, embed_dim=128, depth=4, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(E * M, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, heads, dim_feedforward=embed_dim * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        B, T, E, M = x.shape
        x = x.view(B, T, E * M)
        x = self.input_proj(x)
        x = self.transformer(x)
        x = x.unsqueeze(2).repeat(1, 1, E, 1)  # Broadcast over E for consistency
        return x


# -----------------------------------
# 3. Spatial-only Transformer Encoder
# -----------------------------------
class DrumSpatialTransformer(nn.Module):
    """
    Transformer encoder across instruments per frame.
    - Models global instrument relationships within each time step.
    - No explicit temporal dependencies across time steps.
    """
    def __init__(self, E=9, M=3, embed_dim=128, depth=4, heads=4):
        super().__init__()
        self.input_proj = nn.Linear(M, embed_dim)
        encoder_layer = nn.TransformerEncoderLayer(embed_dim, heads, dim_feedforward=embed_dim * 4, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        B, T, E, M = x.shape
        x = self.input_proj(x)  # B x T x E x D

        x_out = []
        for t in range(T):
            frame = x[:, t, :, :]    # B x E x D
            frame_out = self.transformer(frame)
            x_out.append(frame_out.unsqueeze(1))
        x_out = torch.cat(x_out, dim=1)  # B x T x E x D
        return x_out


# -----------------------------------
# 4. Per-instrument MLP Encoder
# -----------------------------------
class DrumPerInstrumentMLP(nn.Module):
    """
    Simple MLP applied independently to each instrument and time step.
    - No temporal modeling.
    - No instrument interaction.
    - Acts as a feature-wise feedforward baseline.
    """
    def __init__(self, M=3, embed_dim=128, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(M, hidden_dim),
            nn.LeakyReLU(inplace=True),
            nn.Linear(hidden_dim, embed_dim)
        )

    def forward(self, x):
        return self.mlp(x)  # B x T x E x D


# -----------------------------------
# 5. Axial Transformer Encoder
# -----------------------------------
class DrumAxialTransformer(nn.Module):
    def __init__(self, T=33, E=9, M=3, embed_dim=128, depth=6, heads=8, dim_heads=None, reversible=True):
        """
        Axial transformer adapted for drum sequences of shape B x T x E x M.

        This architecture explicitly models both global temporal (T) and instrument (E) relationships 
        while maintaining efficient factorization of attention (axial), 
        which scales better than full 2D self-attention.

        Args:
            T (int): Number of time steps.
            E (int): Number of instruments.
            M (int): Number of expressive features per instrument (e.g., hit, velocity, offset).
            embed_dim (int): Embedding dimension.
            depth (int): Number of axial transformer layers.
            heads (int): Number of attention heads.
            dim_heads (Optional[int]): Head dimension (if None, inferred).
            reversible (bool): Use reversible sequence to save memory.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.dim_heads = dim_heads
        self.axial_pos_shape = (T, E)

        # -------- Initial projection: M channels -> embed_dim --------
        self.input_proj = nn.Conv2d(M, embed_dim, kernel_size=1)
        # This lets us convert per-instrument expressive features into higher-dimensional token embeddings.

        # -------- Axial positional embedding --------
        self.pos_emb = AxialPositionalEmbedding(embed_dim, self.axial_pos_shape, emb_dim_index=1)
        # Provides explicit position information along time (T) and instruments (E) axes separately,
        # which is critical for rhythmic structure and instrument roles.

        # -------- Axial attention permutations --------
        permutations = calculate_permutations(2, emb_dim=1)  # permute across 2 axes: T and E
        # These define the axes (time, instrument) over which we apply separate self-attention passes.

        def get_ff():
            """
            Simple feedforward block: norm → expand channels → activation → project back.
            """
            return nn.Sequential(
                ChanLayerNorm(self.embed_dim),
                nn.Conv2d(self.embed_dim, self.embed_dim * 4, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.embed_dim * 4, self.embed_dim, 3, padding=1)
            )

        layers = nn.ModuleList([])
        for _ in range(depth):
            # -------- Axial attention layers along T and E --------
            attn_layers = nn.ModuleList([
                PermuteToFrom(p, PreNorm(self.embed_dim, SelfAttention(self.embed_dim, self.heads, self.dim_heads)))
                for p in permutations
            ])
            # -------- Local convolutional feedforward layers --------
            conv_layers = nn.ModuleList([get_ff(), get_ff()])
            # Stack axial attention + local conv blocks to capture both long-range dependencies and local interactions
            layers.append(attn_layers)
            layers.append(conv_layers)

        # -------- Reversible sequence for memory efficiency --------
        self.layers = ReversibleSequence(layers) if reversible else Sequential(layers)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
        Returns:
            Tensor of shape (B, T, E, embed_dim)
        """
        B, T, E, M = x.shape

        # -------- Move M to channel dimension --------
        x = x.permute(0, 3, 1, 2)  # (B, M, T, E)

        # -------- Project M channels to embed_dim --------
        x = self.input_proj(x)     # (B, D, T, E)

        # -------- Add axial positional encoding --------
        x = self.pos_emb(x)        # (B, D, T, E)

        # -------- Pass through axial transformer layers --------
        x = self.layers(x)         # (B, D, T, E)

        # -------- Restore shape: (B, T, E, D) --------
        x = x.permute(0, 2, 3, 1)

        return x


# -----------------------------------
# 6. Drum Encoder Wrapper
# -----------------------------------

class DrumEncoderWrapper(nn.Module):
    """
    Wrapper that lets you choose between encoder types easily.

    encoder_type options:
    - "conv": 2D conv baseline (local spatio-temporal).
    - "temporal": temporal transformer baseline (global T, no E).
    - "spatial": spatial transformer baseline (global E, no T).
    - "mlp": per-instrument MLP (no global structure).
    - "axial": axial transformer (global T & E, with positional encoding).
    """
    def __init__(self, encoder_type="axial", T=33, E=9, M=3, embed_dim=128, depth=4, heads=4):
        super().__init__()
        if encoder_type == "conv":
            self.encoder = DrumConvEncoder(M=M, embed_dim=embed_dim, depth=depth)
        elif encoder_type == "temporal":
            self.encoder = DrumTemporalTransformer(E=E, M=M, embed_dim=embed_dim, depth=depth, heads=heads)
        elif encoder_type == "spatial":
            self.encoder = DrumSpatialTransformer(E=E, M=M, embed_dim=embed_dim, depth=depth, heads=heads)
        elif encoder_type == "mlp":
            self.encoder = DrumPerInstrumentMLP(M=M, embed_dim=embed_dim)
        elif encoder_type == "axial":
            self.encoder = DrumAxialTransformer(T=T, E=E, M=M, embed_dim=embed_dim, depth=depth, heads=heads, reversible=False)
        else:
            raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def forward(self, x):
        return self.encoder(x)
