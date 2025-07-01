import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from .sub_modules import *
from vector_quantize_pytorch import VectorQuantize

class DrumAxialTransformer(nn.Module):
    def __init__(self, T=33, E=9, M=3, embed_dim=128, depth=6, heads=8, dim_heads=None, reversible=True):
        """
        Axial transformer adapted for drum sequences of shape B x T x E x M.

        Args:
            in_channels: Number of modifier channels (e.g., 3 for HVO).
            dim: Embedding dimension.
            depth: Number of transformer layers.
            heads: Attention heads.
            dim_heads: Optional head dimension.
            axial_pos_shape: Tuple (T, E) for axial position embedding.
        """
        super().__init__()

        self.embed_dim = embed_dim
        self.depth = depth
        self.heads = heads
        self.dim_heads = dim_heads
        self.axial_pos_shape = (T, E)

        # Initial projection: from (M) to dim
        self.input_proj = nn.Conv2d(M, embed_dim, kernel_size=1)

        # Positional embedding
        self.pos_emb = AxialPositionalEmbedding(embed_dim, self.axial_pos_shape, emb_dim_index=1)

        # Axial attention across T and E (axes 2 and 3)
        permutations = calculate_permutations(2, emb_dim=1)  # across H, W

        def get_ff():
            return nn.Sequential(
                ChanLayerNorm(self.embed_dim),
                nn.Conv2d(self.embed_dim, self.embed_dim * 4, 3, padding=1),
                nn.LeakyReLU(inplace=True),
                nn.Conv2d(self.embed_dim * 4, self.embed_dim, 3, padding=1)
            )

        layers = nn.ModuleList([])
        for _ in range(depth):
            attn_layers = nn.ModuleList([
                PermuteToFrom(p, PreNorm(self.embed_dim, SelfAttention(self.embed_dim, self.heads, self.dim_heads)))
                for p in permutations
            ])
            conv_layers = nn.ModuleList([get_ff(), get_ff()])
            layers.append(attn_layers)
            layers.append(conv_layers)

        self.layers = ReversibleSequence(layers) if reversible else Sequential(layers)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape B x T x E x M
        Returns:
            Output of shape B x T x E x D
        """
        B, T, E, M = x.shape
        x = x.permute(0, 3, 1, 2)  # B x M x T x E
        x = self.input_proj(x)     # B x D x T x E
        x = self.pos_emb(x)
        x = self.layers(x)           # B x D x T x E
        return x.permute(0, 2, 3, 1) # B x T x E x D
    

class LearnableBinsQuantizer(nn.Module):
    def __init__(self, num_bins, min_val=0.0, max_val=1.0):
        super().__init__()
        self.num_bins = num_bins
        self.min_val = min_val
        self.max_val = max_val

        initial_bins = torch.linspace(min_val, max_val, num_bins)
        self.bin_values = nn.Parameter(initial_bins)

    def forward(self, x):
        # Normalize x to [0, 1]
        x_norm = (x - self.min_val) / (self.max_val - self.min_val)
        x_norm = torch.clamp(x_norm, 0, 1)

        # Distance to uniform reference bins [0, 1]
        reference_bins = torch.linspace(0, 1, self.num_bins, device=x.device)
        distances = torch.abs(x_norm.unsqueeze(-1) - reference_bins)

        bin_idx = torch.argmin(distances, dim=-1)  # (...)

        # Use learned bin values
        bin_value = self.bin_values[bin_idx]

        # Straight-through estimator
        out = x + (bin_value - x).detach()
        return out
    

class IQAE(nn.Module):
    """
    ### Input shape:
        x: Tensor of shape (B, T, E, M)
            - B: batch size
            - T: number of time steps
            - E: number of drum instruments
            - M: number of expressive features (e.g., hit, velocity, timing offset)
    """
    def __init__(self, 
                 T=33, E=9, M=3,
                 #z_dim=64,
                 embed_dim=128, encoder_depth=4, encoder_heads=4,
                 decoder_depth=2, decoder_heads=4, 
                 num_buttons=2, num_bins_velocity=8, num_bins_offset=16
    ):
        """
        Args:
            T (int): maximum length this model can generate.
            E (int): Number of drum instruments.
            M (int): Number of expressive features.
            embed_dim (int): Embedding dimension.
            encoder_depth (int): Number of layers in the axial transformer.
            encoder_heads (int): Number of attention heads.
            decoder_depth (int): Number of layers in the decoder.
            decoder_heads (int): Number of attention heads.
            num_buttons (int): Number of buttons.
        """
        super().__init__()
        
        self.T = T
        self.E = E
        self.M = M

        self.embed_dim = embed_dim
        self.encoder_depth = encoder_depth
        self.encoder_heads = encoder_heads
        self.decoder_depth = decoder_depth
        self.num_buttons   = num_buttons

        self.sos_token = nn.Parameter(torch.randn(1, E, M))
        self.pos_emb   = PositionalEncoding(embed_dim, T)
        self.encoder   = DrumAxialTransformer(
                                T=T, E=E, M=M, embed_dim=embed_dim, 
                                depth=encoder_depth, heads=encoder_heads, 
                                dim_heads=None, reversible=False
                        )
        
        #self.z_mu_proj = nn.Linear(embed_dim, z_dim)
        #self.z_logvar_proj = nn.Linear(embed_dim, z_dim)
        self.attn_proj = nn.Linear(embed_dim, 1)
        self.latent_projection = nn.Linear(embed_dim, num_buttons * M) # Pick offset from input features

        self.dec_inp_proj    = nn.Linear(E * M, embed_dim) # (B, T', E*M) -> (B, T', D)
        self.dec_button_proj = nn.Linear(num_buttons * M, embed_dim) # (B, T', num_buttons*M) -> (B, T', D)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=decoder_heads, batch_first=True, norm_first=True),
            num_layers    = decoder_depth,
            norm          = nn.LayerNorm(embed_dim)
        )
        
        self.output_projection = nn.Linear(embed_dim, E * M) # (B, T', D) -> (B, T', E*M)

        # Learned bin quantizers
        self.velocity_quantizer = LearnableBinsQuantizer(num_bins_velocity, min_val=0.0, max_val=1.0)
        self.offset_quantizer = LearnableBinsQuantizer(num_bins_offset, min_val=-0.5, max_val=0.5)

    def aggregate(self, encoded):
        # encoded: (B, T, E, D)
        attn_scores = self.attn_proj(encoded)  # (B, T, E, 1)
        attn_scores = attn_scores.squeeze(-1)  # (B, T, E)
        attn_weights = F.softmax(attn_scores, dim=-1)  # (B, T, E)
        latent = torch.sum(encoded * attn_weights.unsqueeze(-1), dim=2)  # (B, T, D)
        return latent

    def encode(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
        Returns:
            latent: Tensor of shape (B, T, num_buttons, M)
        """
        B, T, E, M = x.shape
        encoded = self.encoder(x)               # (B, T, E, D)
        latent = self.aggregate(encoded)        # (B, T, D)
        latent = self.latent_projection(latent) # (B, T, num_buttons * M)

        # z
        #mu = self.z_mu_proj(latent.mean(dim=1))          # (B, z_dim)
        #logvar = self.z_logvar_proj(latent.mean(dim=1))  # (B, z_dim)
        #std = torch.exp(0.5 * logvar)
        #eps = torch.randn_like(std)
        #z = mu + eps * std  # Reparameterization # (B, z_dim)
        #kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean() # (B)

        button_latent = latent.view(B, T, self.num_buttons, M) # (B, T, num_buttons, M)
        return button_latent
    
    def straight_through_binarize(self, x, threshold=0.5):
        """Applies hard threshold during forward, identity gradient during backward."""
        hard = (x > threshold).float()
        return x + (hard - x).detach()
    

    def make_button_hvo(self, button_latent):
        """
        Args:
            latent: Tensor of shape (B, T, num_buttons, M)
        Returns:
            button_hvo: (B, T, num_buttons, M) — [activation_flag, velocity, offset]
        """
        # Get hits and velocity from latent
        hits_latent = button_latent[:, :, :, 0]       # (B, T, num_buttons)
        velocity_latent = button_latent[:, :, :, 1]   # (B, T, num_buttons)
        offset_latent   = button_latent[:, :, :, 2]   # (B, T, num_buttons)
        hits_latent     = torch.sigmoid(hits_latent)
        hits_latent     = self.straight_through_binarize(hits_latent)
        velocity_latent = (torch.tanh(velocity_latent) + 1.0) / 2.0 # [-1.0, 1.0] -> [0, 1]
        velocity_latent = self.velocity_quantizer(velocity_latent)
        offset_latent   = torch.tanh(offset_latent) * 0.5   # [-1.0, 1.0] -> [-0.5, 0.5]
        offset_latent   = self.offset_quantizer(offset_latent)
        button_hvo      = torch.stack([hits_latent, velocity_latent, offset_latent], dim=-1) # (B, T, num_buttons, M)
        return button_hvo  # (B, T, num_buttons, M)


    def decode(self, input, button_hvo):
        """
        Args:
            input: Tensor of shape  (B, T, E, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            change_mask: Tensor of shape (B, T)
        Returns:
            h: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
        """
        B, T, E, M = input.shape
        num_buttons = self.num_buttons
        
        # Fix device issue: ensure sos_token is on same device as input
        target = torch.cat([self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1), input[:, :-1, :, :]], dim=1) # (B, T', E, M)
        target = self.dec_inp_proj(target.view(B, T, E * M)) # (B, T', D)
        
        target = self.pos_emb(target) # (B, T', D)
        target_causal_mask = CausalMask(target) # (T', T')
        
        memory = self.dec_button_proj(button_hvo.view(B, T, num_buttons * M))  # (B, T', D)
        memory = self.pos_emb(memory) # (B, T', D)
        memory_causal_mask = CausalMask(memory) # (T', T')
        memory_padding_mask = button_hvo[:, :, :, 0].sum(dim=-1) # (B, T')
        memory_padding_mask = (memory_padding_mask == 0).bool()  # (B, T')

        decoder_out = self.decoder(
            tgt = target, 
            memory = memory,
            tgt_mask = target_causal_mask,
            memory_mask = memory_causal_mask,
            memory_key_padding_mask = memory_padding_mask,
            tgt_is_causal = True,
            memory_is_causal = True
        ) # (B, T', D)

        output = self.output_projection(decoder_out) # (B, T', E*M)
        output = output.view(B, T, E, M) # (B, T', E*M) -> (B, T, E, M)

        h_logits = output[:, :, :, 0] # (B, T, E)
        hit_mask = (torch.sigmoid(h_logits) > 0.5).int()    # (B, T, E)
        v = ((torch.tanh(output[:, :, :, 1]) + 1.0) / 2.0) * hit_mask    # (B, T, E)
        o = torch.tanh(output[:, :, :, 2]) * 0.5 * hit_mask # (B, T, E)
        return h_logits, v, o
    
    def sample_z(self, batch_size, device):
        """
        Sample z from standard normal prior
        """
        return torch.randn(batch_size, self.z_dim, device=device)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)

        Returns:
            h: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
            hits_latent: Tensor of shape (B, T)
            button_hits_seq: Tensor of shape (B, T)
            button_velocity_seq: Tensor of shape (B, T)
        """
        # ========== ENCODING ==========
        latent = self.encode(x)             # (B, T, M-1)

        # ========== MAKE BUTTON HVO ==========
        button_hvo = self.make_button_hvo(latent) # (B, T, num_buttons, M)

        # ========== DECODING ==========
        h_logits, v, o = self.decode(x, button_hvo) # (B, T, E), (B, T, E), (B, T, E)

        button_hits = button_hvo[:, :, :, 0] # (B, T, num_buttons)
        button_velocity = button_hvo[:, :, :, 1] # (B, T, num_buttons)
        button_offset   = button_hvo[:, :, :, 2] # (B, T, num_buttons)
        
        # Identify frames with no input hits
        #input_hits_sum = x[:, :, :, 0].sum(dim=-1, keepdim=True) # (B, T, 1)
        #no_input_hit = (input_hits_sum == 0).float()

        # Penalize button hits in those frames
        #button_penalty = (button_hits * no_input_hit)

        no_hit_mask = (button_hits == 0).float()
        velocity_penalty = (button_velocity * no_hit_mask).abs().mean()
        offset_penalty = (button_offset * no_hit_mask).abs().mean()
        
        return h_logits, v, o, latent, button_hvo, velocity_penalty, offset_penalty
    
    def generate(self, button_hvo, max_steps=None):
        """
        Generate a prediction for the input at time t, given input < t, button HVO <= t, and change mask <= t.
        Args:
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            max_steps: int (optional)
        Returns:
            hvo_pred: Tensor of shape (B, T, E, 3)
        """
        B, T, num_buttons, M = button_hvo.shape
        E = self.E
        T_gen = max_steps or T # (B, T, num_buttons, M)

        generated = self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1) # (B, 1, E, M)
        for t in range(T_gen):
            tgt_embed = self.dec_inp_proj(generated.view(B, t + 1, E * M)) # (B, T, D)
            mem_embed = self.dec_button_proj(button_hvo[:, :t + 1].view(B, t + 1, num_buttons * M)) # (B, T, D)

            tgt_embed_pos = self.pos_emb(tgt_embed)
            mem_embed_pos = self.pos_emb(mem_embed)
            tgt_mask = CausalMask(tgt_embed_pos)
            mem_mask = CausalMask(mem_embed_pos)
            mem_pad_mask = (button_hvo[:, :t + 1, :, 0].sum(dim=-1) == 0).bool()
            dec_out = self.decoder(
                tgt = tgt_embed_pos, 
                memory = mem_embed_pos, 
                tgt_mask = tgt_mask, 
                memory_mask = mem_mask, 
                memory_key_padding_mask = mem_pad_mask,
                tgt_is_causal = True,   
                memory_is_causal = True
            ) # (B, t + 1, D)
            output = self.output_projection(dec_out) # (B, t + 1, E * M)
            output = output.view(B, t + 1, E, M)     # (B, t + 1, E, M)
            pred_step = output[:, -1, :, :]          # (B, E, M)
            h_logits = pred_step[:, :, 0]            # (B, E)
            h_pred = (torch.sigmoid(h_logits) > 0.5).int() # (B, E)
            v_pred = ((torch.tanh(pred_step[:, :, 1]) + 1.0) / 2.0) * h_pred # (B, E)
            o_pred = torch.tanh(pred_step[:, :, 2]) * 0.5 * h_pred   # (B, E)
            hvo_pred = torch.stack([h_pred, v_pred, o_pred], dim=-1) # (B, E, 3)
            generated = torch.cat([generated, hvo_pred.unsqueeze(1)], dim=1) # (B, t + 1, E, M)

        return generated

if __name__ == "__main__":
    input_size = (4, 33, 9, 3)
    encoder = IQAE(T=33, E=9, M=3, embed_dim=128, encoder_depth=4, encoder_heads=4, decoder_depth=1, decoder_heads=2, num_buttons=3, num_bins_velocity=8, num_bins_offset=16)
    summary(encoder, input_size=input_size)


        