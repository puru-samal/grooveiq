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

class Codebook(nn.Module):
    def __init__(self, num_codes: int, z_dim: int):
        super().__init__()
        self.num_codes = num_codes
        self.z_dim = z_dim
        self.codebook = nn.Parameter(torch.randn(num_codes, z_dim))

    def forward(self, z_e):
        # z_e: (B, T, z_dim)
        B, T, D = z_e.shape
        z_flat = z_e.view(-1, D)  # (B*T, D)
        dists = torch.cdist(z_flat, self.codebook)  # (B*T, num_codes)
        indices = torch.argmin(dists, dim=1)  # (B*T)
        z_q = self.codebook[indices].view(B, T, D)  # (B, T, D)

        # Straight-through estimator
        z_q_st = z_e + (z_q - z_e).detach()
        return z_q_st, indices.view(B, T)

class VQGrooveIQ(nn.Module):
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
            is_causal: bool = True,
            chunk_size: int = 3,
            button_penalty: int = 1,
            button_dropout: float = 0.5,
            num_codes: int = 1024,
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
            chunk_size (int): Chunk size for pooling.
            p (float): Probability of using z_post instead of z_prior.
            button_penalty (int): Penalty for button hits. 1 : L1, 2 : Group (L1 over T of L2(D))
            button_dropout (float): Dropout probability for button hits.
            num_codes (int): Number of codes in the codebook.
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
        self.chunk_size = chunk_size
        self.button_penalty = button_penalty
        self.threshold = 0.5
        self.num_codes = num_codes

        self.sos_token = nn.Parameter(torch.randn(1, E, M))
        self.pos_emb   = PositionalEncoding(embed_dim, T)
        self.encoder   = DrumAxialTransformer(
                                T=T, E=E, M=M, embed_dim=embed_dim, 
                                depth=encoder_depth, heads=encoder_heads, 
                                dim_heads=None, reversible=False
                        )
        self.encoder_proj = nn.Linear(E * embed_dim, embed_dim)
        self.button_hit_repr  = nn.Linear(embed_dim, num_buttons) # (B, T, D) -> (B, T, num_buttons)
        self.button_dropout   = nn.Dropout(p=button_dropout)
        
        self.button_embed = nn.LSTM(num_buttons, embed_dim, batch_first=True)
        self.button_norm = nn.LayerNorm(embed_dim)
        
        # Posterior network q(z | x, button)
        self.z_mu_proj     = nn.Linear(2 * embed_dim, z_dim)
        self.z_logvar_proj = nn.Linear(2 * embed_dim, z_dim)

        self.codebook = Codebook(num_codes=num_codes, z_dim=z_dim)
        self.code_index_predictor = nn.Sequential(
            nn.LSTM(embed_dim, embed_dim, batch_first=True),
            nn.Linear(embed_dim, self.num_codes)
        )

        self.dec_inp_proj = nn.Linear(E * M, embed_dim) # (B, T', E*M) -> (B, T', D)
        self.dec_z_proj   = nn.Linear(z_dim, embed_dim) # (B, T', z_dim) -> (B, T', D)
        self.dec_btn_proj = nn.Linear(embed_dim, embed_dim) # (B, T', D) -> (B, T', D)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=decoder_heads, batch_first=True, norm_first=True),
            num_layers    = decoder_depth,
            norm          = nn.LayerNorm(embed_dim)
        )
        
        self.output_projection = nn.Linear(embed_dim, E * M) # (B, T', D) -> (B, T', E*M)

    def chunk_and_expand(self, tensor):
        """
        Args:
            tensor: Tensor of shape (B, T, D)
            chunk_size: int
        Returns:
            expanded: Tensor of shape (B, T_ * chunk_size, D)
        """
        B, T, D = tensor.shape
        T_ = T // self.chunk_size
        pooled = tensor[:, :T_ * self.chunk_size].view(B, T_, self.chunk_size, D).mean(dim=2) # (B, T_, D)
        expanded = pooled.unsqueeze(2).repeat(1, 1, self.chunk_size, 1).view(B, T_ * self.chunk_size, D) # (B, T_ * chunk_size, D)
        return expanded

    def encode(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
        Returns:
            encoded: Tensor of shape (B, T, D)
            button_repr: Tensor of shape (B, T, num_buttons)
        """
        B, T, E, M = x.shape
        encoded = self.encoder(x)                    # (B, T, E, D)
        encoded = self.encoder_proj(encoded.reshape(B, T, -1)) # (B, T, D)
        button_repr = self.button_hit_repr(encoded) # (B, T, num_buttons)
        return encoded, button_repr
    
    def calculate_button_penalty(self, button_repr):
        """
        Args:
            button_repr: Tensor of shape (B, T, num_buttons)
        Returns:
            button_penalty: Tensor of shape (B, T)
        """
        if self.button_penalty == 1: # L1
            return button_repr.abs().sum(dim=2).mean()
        elif self.button_penalty == 2: # Group (L1 over T of L2(D))
            return torch.norm(button_repr, dim=2).mean()
        else:
            raise ValueError(f"Invalid button penalty: {self.button_penalty}")
    
    def straight_through_binarize(self, x, threshold=0.5):
        """Applies hard threshold during forward, identity gradient during backward."""
        hard = (x > threshold).float()
        return x + (hard - x).detach()
    
    def make_button_hits(self, button_repr):
        """
        Args:
            button_repr: Tensor of shape (B, T, num_buttons)
        Returns:
            button_hits: Tensor of shape (B, T, num_buttons)
        """
        button_hits = torch.sigmoid(button_repr)
        return self.straight_through_binarize(button_hits)
    
    def make_button_embed(self, button_hits):
        """
        Args:
            button_hits: Tensor of shape (B, T, num_buttons)
        Returns:
            button_embed: Tensor of shape (B, T, D)
        """
        button_embed, _ = self.button_embed(button_hits) # (B, T, D)
        button_embed = self.button_norm(button_embed)    # (B, T, D)
        return button_embed

    def make_z_post(self, button_embed, encoded):
        """
        Args:
            button_embed: Tensor of shape (B, T, D)
        Returns:
            z_post: Tensor of shape (B, T, z_dim)
            mu: Tensor of shape (B, T, z_dim)
            logvar: Tensor of shape (B, T, z_dim)
        """
        enc_pooled = self.chunk_and_expand(encoded)      # (B, T_, D)
        btn_pooled = self.chunk_and_expand(button_embed) # (B, T_, D)
        
        z_e = torch.cat([enc_pooled, btn_pooled], dim=-1)  # (B, T, 2*D)
        z_e = self.z_mu_proj(z_e)                          # (B, T, z_dim) 
        z_q, code_indices = self.codebook(z_e)             # (B, T, z_dim), (B, T)
        return z_q, z_e, code_indices
    
    def predict_code_indices(self, button_embed):
        """
        Args:
            button_embed: Tensor of shape (B, T, D)
        Returns:
            logits: Tensor of shape (B, T, num_codes)
        """
        pred_out, _ = self.code_index_predictor[0](button_embed)  # LSTM
        logits = self.code_index_predictor[1](pred_out)           # (B, T, num_codes)
        return logits

    def decode(self, input, z, button_embed):
        """
        Args:
            input: Tensor of shape  (B, T, E, M)
            z: Tensor of shape (B, T, z_dim)
            button_embed: Tensor of shape (B, T, D)
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
        
        # Memory
        memory = self.dec_z_proj(z) + self.dec_btn_proj(button_embed)  # (B, T', D)
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

    def forward(self, x, button_hits=None):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
            button_hits: Tensor of shape (B, T, num_buttons) (optional)
        Returns:
            h_logits: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
            attn_weights: Dictionary of attention weights
            kl_loss: Tensor of shape (B)
        """
        # Encode
        encoded, button_repr = self.encode(x) # (B, T, D), (B, T, num_buttons)
        button_penalty = self.calculate_button_penalty(button_repr)
        
        # Button processing
        if button_hits is None:
            button_hits = self.make_button_hits(button_repr) # (B, T, num_buttons)

        button_embed = self.make_button_embed(button_hits) # (B, T, D)
        button_embed = self.button_dropout(button_embed) # (B, T, D)
        
        z_q, z_e, code_indices = self.make_z_post(button_embed, encoded) # (B, T, z_dim), (B, T, z_dim), (B, T)
        z = z_q # quantized embedding
    
        # VQ losses
        commitment_loss = F.mse_loss(z_e.detach(), z_q)
        codebook_loss = F.mse_loss(z_e, z_q.detach())
        vq_loss = codebook_loss + 0.25 * commitment_loss

        # Predict logits from button_embed for code indices
        code_pred_logits = self.predict_code_indices(button_embed)
        ce_loss = F.cross_entropy(code_pred_logits.view(-1, self.num_codes), code_indices.view(-1)) 

        # Decode
        h_logits, v, o, attn_weights = self.decode(x, z, button_embed) # (B, T, E), (B, T, E), (B, T, E)

        # Button HVO (for visualization)
        button_hvo = torch.cat([button_hits.unsqueeze(-1),  torch.zeros_like(button_hits).unsqueeze(-1).repeat(1, 1, 1, self.M - 1)], dim=-1) # (B, T, num_buttons, M)
        
        return {
                'h_logits': h_logits, 
                'v': v, 
                'o': o,
                'button_hvo': button_hvo,
                'attn_weights': attn_weights, 
                'vq_loss': vq_loss,
                'ce_loss': ce_loss,
                'button_penalty': button_penalty,
                'code_indices': code_indices,
            }
    
    def generate(self, button_embed, z=None, max_steps=None, threshold=None):
        """
        Generate a prediction for the input at time t, given input < t, button HVO <= t, and latent vector z.
        Args:
            z: Tensor of shape (B, T, z_dim)
            button_embed: Tensor of shape (B, T, D)
            max_steps: int (optional)
        Returns:
            hvo_pred: Tensor of shape (B, T, E, 3)
            hit_logits: Tensor of shape (B, T, E) for threshold calculation
        """
        B, T, _ = button_embed.shape
        E = self.E
        T_gen = int(max_steps or T)

        if threshold is None:
            threshold = self.threshold

        generated = self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1) # (B, 1, E, M)
        hit_probs = []
        if z is None:
            pred_code_indices = self.predict_code_indices(button_embed).argmax(dim=-1) # (B, T)
            z = self.codebook.codebook[pred_code_indices] # (B, T, z_dim)
        
        for t in range(T_gen):

            # Target
            tgt_embed = self.dec_inp_proj(generated.view(B, t + 1, E * self.M)) # (B, T, D)
            tgt_embed_pos = self.pos_emb(tgt_embed)
            
            # Memory
            if self.is_causal:
                mem_embed = self.dec_z_proj(z[:, :t + 1]) + self.dec_btn_proj(button_embed[:, :t + 1]) # (B, T, D)
            else:
                mem_embed = self.dec_z_proj(z) + self.dec_btn_proj(button_embed) # (B, T, D)
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
    input_size = [(4, 33, 9, 3)]
    model = VQGrooveIQ(
        T=33, E=9, M=3, 
        z_dim=32, embed_dim=128, 
        encoder_depth=2, encoder_heads=4, 
        decoder_depth=2, decoder_heads=2, 
        num_buttons=2, is_causal=True,
        num_codes=64,
    )
    summary(model, input_size=input_size)
    



        