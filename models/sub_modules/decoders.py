import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal

# -----------------------------------
# Decoder variants
# -----------------------------------

class ARMLPDecoder(nn.Module):
    """Autoregressive MLP decoder"""
    def __init__(self, embed_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, tgt_embed, memory_embed):
        x = torch.cat([tgt_embed, memory_embed], dim=-1)
        return self.mlp(x)

class ARGRUDecoder(nn.Module):
    """Autoregressive GRU decoder"""
    def __init__(self, embed_dim, hidden_dim=256, output_dim=None):
        super().__init__()
        self.gru = nn.GRU(embed_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, output_dim)

    def forward(self, tgt_embed, memory_embed):
        x = torch.cat([tgt_embed, memory_embed], dim=-1)
        out, _ = self.gru(x)
        return self.fc_out(out)

class ARConvDecoder(nn.Module):
    """Autoregressive 1D Conv decoder"""
    def __init__(self, embed_dim, output_dim, hidden_dim=256, kernel_size=3):
        super().__init__()
        self.conv1 = nn.Conv1d(embed_dim * 2, hidden_dim, kernel_size, padding=kernel_size - 1)
        self.conv2 = nn.Conv1d(hidden_dim, output_dim, kernel_size, padding=kernel_size - 1)
    
    def forward(self, tgt_embed, memory_embed):
        x = torch.cat([tgt_embed, memory_embed], dim=-1).transpose(1, 2)  # (B, D*2, T)
        x = F.relu(self.conv1(x))[:, :, :-2]  # Remove future padding
        x = self.conv2(x)[:, :, :-2]
        return x.transpose(1, 2)  # (B, T, output_dim)

# -----------------------------------
# Drum Decoder Wrapper
# -----------------------------------

class DrumDecoderWrapper(nn.Module):
    """
    Flexible wrapper for different autoregressive decoder types.

    decoder_type options:
    - "transformer": classic transformer decoder (full attention, global context)
    - "gru": GRU-based decoder (temporal, recurrent)
    - "mlp": simple causal MLP per time step
    - "conv": causal 1D convolutional decoder
    """
    def __init__(self, decoder_type : Literal["transformer", "gru", "mlp", "conv"] = "transformer", 
                 embed_dim : int =128, output_dim : int =512, 
                 hidden_dim : int =256, depth : int = 2, heads : int = 4):
        super().__init__()

        self.decoder_type = decoder_type
        self.output_dim = output_dim

        if decoder_type == "transformer":
            self.decoder = nn.TransformerDecoder(
                decoder_layer=nn.TransformerDecoderLayer(d_model=embed_dim, nhead=heads, batch_first=True, norm_first=True),
                num_layers=depth,
                norm=nn.LayerNorm(embed_dim)
            )
            self.output_projection = nn.Linear(embed_dim, output_dim)

        elif decoder_type == "gru":
            self.decoder = ARGRUDecoder(embed_dim, hidden_dim=hidden_dim, output_dim=output_dim)
            self.output_projection = nn.Identity()

        elif decoder_type == "mlp":
            self.decoder = ARMLPDecoder(embed_dim, output_dim, hidden_dim=hidden_dim)
            self.output_projection = nn.Identity()

        elif decoder_type == "conv":
            self.decoder = ARConvDecoder(embed_dim, output_dim, hidden_dim=hidden_dim)
            self.output_projection = nn.Identity()

        else:
            raise ValueError(f"Unknown decoder_type: {decoder_type}")

    def forward(self, tgt_embed, memory_embed, tgt_mask=None, memory_mask=None, memory_key_padding_mask=None):
        if self.decoder_type == "transformer":
            decoder_out = self.decoder(
                tgt=tgt_embed, 
                memory=memory_embed, 
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                tgt_is_causal=True,
                memory_is_causal=True
            )
            output = self.output_projection(decoder_out)
        else:
            output = self.decoder(tgt_embed, memory_embed)
        return output
