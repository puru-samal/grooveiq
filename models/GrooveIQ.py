import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Literal
from .sub_modules import PositionalEncoding
from .sub_modules.encoders import DrumEncoderWrapper
from .sub_modules.decoders import DrumDecoderWrapper
from .sub_modules.bin_quantizer import LearnableBinsQuantizer
from torchinfo import summary

# -----------------------------------
# GrooveIQ Main Module
# -----------------------------------

class GrooveIQ(nn.Module):
    def __init__(self, 
                 T : int = 33,  # Time steps
                 E : int =  9,  # Instruments
                 M : int =  3,  # Buttons
                 embed_dim : int = 128, # Embedding dimension
                 encoder_type : Literal["conv", "temporal", "spatial", "mlp", "axial"] = "axial",
                 encoder_depth : int = 4, 
                 encoder_heads : int = 4,
                 decoder_type : Literal["transformer", "mlp", "gru", "conv"] = "transformer", 
                 decoder_depth : int = 2, 
                 decoder_heads : int = 4,
                 num_buttons :   int = 2,     # Number of buttons
                 num_bins_velocity : int = 8, # Number of bins for velocity
                 num_bins_offset : int = 16,  # Number of bins for offset
    ):
        super().__init__()
        self.T = T
        self.E = E
        self.M = M
        self.embed_dim = embed_dim
        self.num_buttons = num_buttons

        # --- Encoder ---
        self.sos_token = nn.Parameter(torch.randn(1, E, M))
        self.pos_emb = PositionalEncoding(embed_dim, T)
        
        self.encoder = DrumEncoderWrapper(
            encoder_type = encoder_type,
            T = T,
            E = E,
            M = M,
            embed_dim = embed_dim,
            depth = encoder_depth,
            heads = encoder_heads
        )
        
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.latent_projection = nn.Linear(embed_dim, num_buttons * M)

        # --- Decoder ---
        self.dec_inp_proj = nn.Linear(E * M, embed_dim)
        self.dec_button_proj = nn.Linear(num_buttons * M, embed_dim)

        self.decoder = DrumDecoderWrapper(
            decoder_type = decoder_type,
            embed_dim    = embed_dim,
            output_dim   = E * M,
            hidden_dim   = embed_dim * 2,
            depth        = decoder_depth,
            heads        = decoder_heads 
        )

        # --- Quantizers ---
        self.velocity_quantizer = LearnableBinsQuantizer(num_bins_velocity, min_val=0.0, max_val=1.0)
        self.offset_quantizer = LearnableBinsQuantizer(num_bins_offset, min_val=-0.5, max_val=0.5)

    def aggregate(self, encoded):
        """
        Args:
            encoded: Tensor of shape (B, T, E, D)
        Returns:
            latent: Tensor of shape (B, T, D)
        """
        # encoded: (B, T, E, D)
        attn_scores = self.attn_proj(encoded).squeeze(-1) # (B, T, E)
        attn_weights = F.softmax(attn_scores, dim=-1)     # (B, T, E)
        latent = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=2) # (B, T, D)
        return latent

    def encode(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
        Returns:
            latent: Tensor of shape (B, T, num_buttons, M)
        """
        B, T, E, M = x.shape
        encoded = self.encoder(x)        # (B, T, E, D)
        latent = self.aggregate(encoded) # (B, T, D)
        latent = self.latent_projection(latent).view(B, T, self.num_buttons, M) # (B, T, num_buttons, M)
        return latent

    def straight_through_binarize(self, x, threshold=0.5):
        """
        Applies hard threshold during forward, identity gradient during backward.
        """
        hard = (x > threshold).float()
        return x + (hard - x).detach()

    def make_button_hvo(self, latent):
        """
        Args:
            latent: Tensor of shape (B, T, num_buttons, M)
        Returns:
            button_hvo: (B, T, num_buttons, M) — [hits, velocity, offset]
        """
        hits_latent     = torch.sigmoid(latent[:, :, :, 0])            # (B, T, num_buttons)
        velocity_latent = (torch.tanh(latent[:, :, :, 1]) + 1.0) / 2.0 # (B, T, num_buttons)
        offset_latent   = torch.tanh(latent[:, :, :, 2]) * 0.5         # (B, T, num_buttons)
        
        # Quantize
        hits_latent     = self.straight_through_binarize(hits_latent)  # (B, T, num_buttons)
        velocity_latent = self.velocity_quantizer(velocity_latent)     # (B, T, num_buttons)
        offset_latent   = self.offset_quantizer(offset_latent)
        return torch.stack([hits_latent, velocity_latent, offset_latent], dim=-1) # (B, T, num_buttons, M)

    def decode(self, input, button_hvo):
        """
        Args:
            input: Tensor of shape (B, T, E, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
        Returns:
            h: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
        """
        B, T, E, M = input.shape
        num_buttons = self.num_buttons

        target = torch.cat([self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1), input[:, :-1]], dim=1) # (B, T, E, M)
        tgt_embed = self.dec_inp_proj(target.view(B, T, E * M)) # (B, T, D)
        mem_embed = self.dec_button_proj(button_hvo.view(B, T, num_buttons * M)) # (B, T, D)

        tgt_mask = torch.triu(torch.ones(T, T, device=input.device), diagonal=1).bool() if self.decoder.decoder_type == "transformer" else None # (T, T)
        mem_mask = torch.triu(torch.ones(T, T, device=input.device), diagonal=1).bool() if self.decoder.decoder_type == "transformer" else None
        mem_pad_mask = (button_hvo[:, :, :, 0].sum(dim=-1) == 0).bool() if self.decoder.decoder_type == "transformer" else None

        dec_out = self.decoder(
            tgt_embed, mem_embed, tgt_mask=tgt_mask, memory_mask=mem_mask, memory_key_padding_mask=mem_pad_mask
        ) # (B, T, E * M)

        output = dec_out.view(B, T, E, M) 
        h_logits = output[:, :, :, 0] 
        hit_mask = (torch.sigmoid(h_logits) > 0.5).int()
        v = ((torch.tanh(output[:, :, :, 1]) + 1.0) / 2.0) * hit_mask
        o = torch.tanh(output[:, :, :, 2]) * 0.5 * hit_mask
        return h_logits, v, o

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
        Returns:
            h_logits: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
            latent: Tensor of shape (B, T, num_buttons, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            velocity_penalty: Tensor of shape (1) (discourages ghost articulation)
            offset_penalty: Tensor of shape (1) (discourages ghost articulation)
        """
        # ========== ENCODING ==========
        latent = self.encode(x)             # (B, T, M-1)

        # ========== MAKE BUTTON HVO ==========
        button_hvo = self.make_button_hvo(latent) # (B, T, num_buttons, M)

        # ========== DECODING ==========
        h_logits, v, o = self.decode(x, button_hvo) # (B, T, E)
        button_hits = button_hvo[:, :, :, 0]        # (B, T, num_buttons)
        button_velocity = button_hvo[:, :, :, 1]    # (B, T, num_buttons)
        button_offset = button_hvo[:, :, :, 2]      # (B, T, num_buttons)

        # ========== CALCULATE PENALTIES ==========
        no_hit_mask = (button_hits == 0).float()
        velocity_penalty = (button_velocity * no_hit_mask).abs().mean()
        offset_penalty = (button_offset * no_hit_mask).abs().mean()

        return h_logits, v, o, latent, button_hvo, velocity_penalty, offset_penalty

    def generate(self, button_hvo, max_steps=None):
        """
        Generate a prediction at time t, given predictions < t and button HVO <= t.
        Args:
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            max_steps: int (optional)
        Returns:
            hvo_pred: Tensor of shape (B, T, E, M)
        """
        B, T, num_buttons, M = button_hvo.shape 
        E = self.E
        T_gen = max_steps or T # (B, T, num_buttons, M)

        generated = self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1) # (B, 1, E, M)
        for t in range(T_gen):
            tgt_embed = self.dec_inp_proj(generated.view(B, t + 1, E * M)) # (B, T, D)
            mem_embed = self.dec_button_proj(button_hvo[:, :t + 1].view(B, t + 1, num_buttons * M)) # (B, T, D)

            if self.decoder.decoder_type == "transformer":
                tgt_embed_pos = self.pos_emb(tgt_embed)
                mem_embed_pos = self.pos_emb(mem_embed)
                tgt_mask = torch.triu(torch.ones(t + 1, t + 1, device=button_hvo.device), diagonal=1).bool()
                mem_mask = torch.triu(torch.ones(t + 1, t + 1, device=button_hvo.device), diagonal=1).bool()
                mem_pad_mask = (button_hvo[:, :t + 1, :, 0].sum(dim=-1) == 0).bool()
                dec_out = self.decoder(
                    tgt_embed_pos, mem_embed_pos, tgt_mask=tgt_mask, memory_mask=mem_mask, memory_key_padding_mask=mem_pad_mask
                )
                out_step = dec_out[:, -1, :] # (B, E * M)
            else:
                mem_pad_mask = (button_hvo[:, :t + 1, :, 0].sum(dim=-1) == 0).int() # (B, T)
                dec_out = self.decoder(tgt_embed, mem_embed * (~mem_pad_mask).unsqueeze(-1)) # (B, E * M)
                out_step = dec_out[:, -1, :] # (B, E * M)

            pred_step = out_step.view(B, E, M)
            h_logits = pred_step[:, :, 0]
            h_pred = (torch.sigmoid(h_logits) > 0.5).int()
            v_pred = ((torch.tanh(pred_step[:, :, 1]) + 1.0) / 2.0) * h_pred
            o_pred = torch.tanh(pred_step[:, :, 2]) * 0.5 * h_pred
            hvo_pred_step = torch.stack([h_pred, v_pred, o_pred], dim=-1) # (B, E, M)
            generated = torch.cat([generated, hvo_pred_step.unsqueeze(1)], dim=1) # (B, T, E, M)

        return generated

if __name__ == "__main__":
    input_size = (4, 33, 9, 3)
    model = GrooveIQ(
        T=33, E=9, M=3, 
        embed_dim=128, 
        encoder_type="axial", encoder_depth=3, encoder_heads=2, 
        decoder_type="transformer", decoder_depth=1, decoder_heads=2, 
        num_buttons=3, num_bins_velocity=8, num_bins_offset=16
    )
    summary(model, input_size=input_size)