import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from typing import Literal
from torch.autograd import Function
from .sub_modules import (
    DrumAxialTransformer, 
    PositionalEncoding, 
    CausalMask, 
    TransformerDecoderLayer
)

class GrooveIQ2(nn.Module):
    """
    ### Input shape:
        x: Tensor of shape (B, T, E, M)
            - B: batch sizeyes
            - T: number of time steps
            - E: number of drum instruments
            - M: number of expressive features (e.g., hit, velocity, timing offset)
    """
    def __init__(
            self, 
            T=33, E=9, M=3,
            z_dim=64, embed_dim=128, 
            encoder_depth=4, encoder_heads=4,
            decoder_depth=2, decoder_heads=4, 
            num_buttons=2,
            is_causal: bool = True
    ):
        """
        Args:
            T (int): maximum length this model can generate.
            E (int): Number of drum instruments.
            M (int): Number of expressive features.
            z_dim (int): Dimension of the latent vector.
            embed_dim (int): Embedding dimension.
            encoder_depth (int): Number of layers in the axial transformer.
            encoder_heads (int): Number of attention heads.
            decoder_depth (int): Number of layers in the decoder.
            decoder_heads (int): Number of attention heads.
            num_buttons (int): Number of buttons.
            is_causal (bool): Whether to use causal attention.
        """
        super().__init__()
        
        self.T = T
        self.E = E
        self.M = M
        self.z_dim = z_dim
        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.decoder_depth = decoder_depth
        self.num_buttons   = num_buttons
        self.is_causal = is_causal

        self.sos_token = nn.Parameter(torch.randn(1, E, M))
        self.pos_emb   = PositionalEncoding(embed_dim, T)
        self.encoder   = DrumAxialTransformer(
                                T=T, E=E, M=M, embed_dim=embed_dim, 
                                depth=encoder_depth, heads=encoder_heads, 
                                dim_heads=None, reversible=False
                        )
        
        self.button_embed = nn.GRU(num_buttons, embed_dim, batch_first=True)
        
        # Posterior network q(z | x, button_mask)
        self.z_mu_proj     = nn.Linear(2 * embed_dim, z_dim)
        self.z_logvar_proj = nn.Linear(2 * embed_dim, z_dim)

        # Prior p(z|mask) via RNN
        self.latent_prior    = nn.GRU(embed_dim, z_dim, batch_first=True)
        self.z_prior_mu      = nn.Linear(z_dim, z_dim)
        self.z_prior_logvar  = nn.Linear(z_dim, z_dim)
        self.attn_proj_instr = nn.Linear(embed_dim, 1)

        self.dec_inp_proj      = nn.Linear(E * M, embed_dim) # (B, T', E*M) -> (B, T', D)
        self.dec_z_proj   = nn.Linear(z_dim, embed_dim) # (B, T', z_dim) -> (B, T', D)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=decoder_heads, batch_first=True, norm_first=True),
            num_layers    = decoder_depth,
            norm          = nn.LayerNorm(embed_dim)
        )
        
        self.output_projection = nn.Linear(embed_dim, E * M) # (B, T', D) -> (B, T', E*M)

    def aggregate_instrument(self, encoded):
        # (B, T, E, D) -> (B, T, D)
        attn_scores = self.attn_proj_instr(encoded).squeeze(-1) # (B, T, E)
        attn_weights = F.softmax(attn_scores, dim=-1)           # (B, T, E)
        latent = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=2)  # (B, T, D)
        return latent

    def encode(self, x, button_hits):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
            button_hits: Binary hit mask of shape (B, T, num_buttons)
        Returns:
            z: Tensor of shape (B, T, z_dim)
            mu: Tensor of shape (B, T, z_dim)
            kl_loss: scalar
        """
        B, T, E, M = x.shape
        encoded = self.encoder(x)                    # (B, T, E, D)
        encoded = self.aggregate_instrument(encoded) # (B, T, D)
    
        # Posterior network q(z_t | x_t)
        button_embed, _ = self.button_embed(button_hits) # (B, T, D)
        enc_mask_cat = torch.cat([encoded, button_embed], dim=-1)    # (B, T, 2 * D)
        mu = self.z_mu_proj(enc_mask_cat)                            # (B, T, z_dim)
        logvar = self.z_logvar_proj(enc_mask_cat)                    # (B, T, z_dim)
        std = torch.exp(0.5 * logvar)                                # (B, T, z_dim)
        eps = torch.randn_like(std)                                  # (B, T, z_dim)
        z_post = mu + eps * std                                      # (B, T, z_dim)

        # Prior from mask only
        mask_embed, _ = self.latent_prior(button_embed) # (B, T, z_dim)
        mu_prior = self.z_prior_mu(mask_embed)         # (B, T, z_dim)
        logvar_prior = self.z_prior_logvar(mask_embed) # (B, T, z_dim)

        kl = 0.5 * torch.sum(
            logvar_prior - logvar + (logvar.exp() + (mu - mu_prior)**2) / logvar_prior.exp() - 1
        )

        return z_post, mu, kl

    def decode(self, input, z):
        """
        Args:
            input: Tensor of shape  (B, T, E, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            z: Tensor of shape (B, T, z_dim)
        Returns:
            h: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
        """
        B, T, E, M = input.shape
        num_buttons = self.num_buttons

        # Target
        target = torch.cat([self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1), input[:, :-1, :, :]], dim=1) # (B, T', E, M)
        target = self.dec_inp_proj(target.view(B, T, E * M)) # (B, T', D)
        target = self.pos_emb(target) # (B, T', D)
        target_causal_mask = CausalMask(target) # (T', T')
        
        memory = self.dec_z_proj(z)  # (B, T', D)
        memory = self.pos_emb(memory)     # (B, T', D)
        memory_causal_mask = CausalMask(memory) if self.is_causal else None # (T', T')

        decoder_out = self.decoder(
            tgt = target, 
            memory = memory,
            tgt_mask = target_causal_mask,
            memory_mask = memory_causal_mask,
            tgt_is_causal = True,
            memory_is_causal = self.is_causal
        ) # (B, T', D)

        output = self.output_projection(decoder_out) # (B, T', E*M)
        output = output.view(B, T, E, M) # (B, T', E*M) -> (B, T, E, M)
        h_logits = output[:, :, :, 0] # (B, T, E)
        v = ((torch.tanh(output[:, :, :, 1]) + 1.0) / 2.0)# (B, T, E)
        o = torch.tanh(output[:, :, :, 2]) * 0.5 # (B, T, E)

        attn_weights = {
            'self_attn': self.decoder.layers[0].self_attn_weights,   # first layer (B, T', T')
            'cross_attn': self.decoder.layers[-1].cross_attn_weights # last layer (B, T', T')
        }
        return h_logits, v, o, attn_weights
    
    def sample_z_from_button_hits(self, button_hits):
        """
        Args:
            button_hits: Tensor of shape (B, T, num_buttons)
        Returns:
            z: Tensor of shape (B, T, z_dim)
        """
        button_hits = button_hits.float()                # (B, T, num_buttons)
        button_embed, _ = self.button_embed(button_hits) # (B, T, D)
        mask_embed, _ = self.latent_prior(button_embed)          # (B, T, z_dim)
        mu_prior = self.z_prior_mu(mask_embed)    # (B, T, z_dim)
        logvar_prior = self.z_prior_logvar(mask_embed)  # (B, T, z_dim)
        std = torch.exp(0.5 * logvar_prior)             # (B, T, z_dim)
        eps = torch.randn_like(std)                     # (B, T, z_dim)
        z = mu_prior + eps * std                        # (B, T, z_dim)
        return z

    def forward(self, x, button_hits):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
            button_hits: Tensor of shape (B, T, num_buttons)
        Returns:
            h_logits: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
            attn_weights: Dictionary of attention weights
            kl_loss: Tensor of shape (B)
        """
        # ========== ENCODING ==========
        z, mu, kl_loss = self.encode(x, button_hits) # (B, T, z_dim), (B, T, z_dim), (B)

        # ========== DECODING ==========
        h_logits, v, o, attn_weights = self.decode(x, z) # (B, T, E), (B, T, E), (B, T, E)

        return {
                'h_logits': h_logits, 
                'v': v, 
                'o': o, 
                'attn_weights': attn_weights, 
                'kl_loss': kl_loss, 
            }
    
    def generate(self, button_hits, z=None, max_steps=None, threshold=0.5):
        """
        Generate a prediction for the input at time t, given input < t, button HVO <= t, and latent vector z.
        Args:
            button_hits: Tensor of shape (B, T, num_buttons)
            z: Tensor of shape (B, z_dim)
            max_steps: int (optional)
        Returns:
            hvo_pred: Tensor of shape (B, T, E, 3)
            hit_logits: Tensor of shape (B, T, E) for threshold calculation
        """
        B, T, num_buttons = button_hits.shape
        E = self.E
        T_gen = int(max_steps or T)

        generated = self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1) # (B, 1, E, M)
        hit_probs = []
        
        if z is None:
            z = self.sample_z_from_button_hits(button_hits) # (B, T, z_dim)
        
        for t in range(T_gen):

            # Target
            tgt_embed = self.dec_inp_proj(generated.view(B, t + 1, E * self.M)) # (B, T, D)
            tgt_embed_pos = self.pos_emb(tgt_embed)
            
            # Memory
            if self.is_causal:
                mem_embed = self.dec_z_proj(z[:, :t + 1]) # (B, T, D)
            else:
                mem_embed = self.dec_z_proj(z) # (B, T, D)
            mem_embed_pos = self.pos_emb(mem_embed)
            
            # Mask
            tgt_mask = CausalMask(tgt_embed_pos)
            mem_mask = CausalMask(mem_embed_pos) if self.is_causal else None
           
            dec_out = self.decoder(
                tgt = tgt_embed_pos, 
                memory = mem_embed_pos, 
                tgt_mask = tgt_mask, 
                memory_mask = mem_mask, 
                tgt_is_causal = True,   
                memory_is_causal = self.is_causal
            ) # (B, t + 1, D)

            # Output
            output = self.output_projection(dec_out) # (B, t + 1, E * M)
            output = output.view(B, t + 1, E, -1)     # (B, t + 1, E, M)

            # Predict
            pred_step = output[:, -1, :, :]          # (B, E, M)
            h_logits = pred_step[:, :, 0]            # (B, E)
            h_prob = torch.sigmoid(h_logits)
            hit_probs.append(h_prob)

            # Predict
            h_pred = (h_prob > threshold).int() # (B, E)
            v_pred = ((torch.tanh(pred_step[:, :, 1]) + 1.0) / 2.0) * h_pred 
            o_pred = torch.tanh(pred_step[:, :, 2]) * 0.5 * h_pred    
            hvo_pred  = torch.stack([h_pred, v_pred, o_pred], dim=-1) # (B, E, 3)
            generated = torch.cat([generated, hvo_pred.unsqueeze(1)], dim=1) # (B, t + 1, E, M)

        return generated, torch.stack(hit_probs, dim=1) # (B, T, E)

if __name__ == "__main__":
    input_size = [(4, 33, 9, 3), (4, 33, 2)]
    encoder = GrooveIQ2(
        T=33, E=9, M=3, z_dim=64,
        embed_dim=128, encoder_depth=2, encoder_heads=4, 
        decoder_depth=2, decoder_heads=2, 
        num_buttons=2, is_causal=True
    )
    summary(encoder, input_size=input_size)


        