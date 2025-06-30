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
                 embed_dim=128, encoder_depth=4, encoder_heads=4,
                 decoder_depth=2, decoder_heads=4, num_buttons=2
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
        
        self.pool = nn.AdaptiveAvgPool2d((M, M))
        self.latent_projection = nn.Linear(M * M, M - 1) # Pick offset from input features
        self.hit_quantizer = IntegerQuantizer(num_bins=num_buttons)

        self.dec_inp_proj    = nn.Linear(E * M, embed_dim) # (B, T', E*M) -> (B, T', D)
        self.dec_button_proj = nn.Linear(num_buttons * M, embed_dim) # (B, T', num_buttons*M) -> (B, T', D)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer = nn.TransformerDecoderLayer(d_model=embed_dim, nhead=decoder_heads, batch_first=True, norm_first=True),
            num_layers    = decoder_depth,
            norm          = nn.LayerNorm(embed_dim)
        )
        
        self.output_projection = nn.Linear(embed_dim, E * M) # (B, T', D) -> (B, T', E*M)

    def encode(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
        Returns:
            hits_latent: Tensor of shape (B, T)
            velocity_latent: Tensor of shape (B, T)
        """
        B, T, E, M = x.shape
        encoded = self.encoder(x)           # (B, T, E, D)
        x_context = self.pool(encoded)      # (B, T, M, M)
        latent = self.latent_projection(x_context.view(B, T, M * M)) # (B, T, M-1)
        return latent
    

    def quantize(self, hits_latent):
        """
        Args:
            hits_latent: Tensor of shape (B, T)
        Returns:
            Tensor of shape (B, T, 1)
        """
        return self.hit_quantizer(hits_latent)
    

    def pick_offset(self, input, change_mask):
        """
        Pick a single offset per time step based on the instrument (E) with
        the highest velocity, only if change_mask is True at that time step.

        Args:
            input: Tensor of shape (B, T, E, M)
                where M = [hit, velocity, offset]
            change_mask: Tensor of shape (B, T), dtype=torch.bool

        Returns:
            Tensor of shape (B, T) — picked offset per time step
        """
        velocity = input[..., 1]       # (B, T, E)
        offset   = input[..., 2]       # (B, T, E)

        # Find the instrument with the highest velocity per time step
        best_idx = velocity.argmax(dim=-1)  # (B, T) — index into E

        # Gather the corresponding offsets
        selected_offset = torch.gather(offset, dim=2, index=best_idx.unsqueeze(-1)).squeeze(-1)  # (B, T)

        # Apply change_mask — zero out where no change
        selected_offset = selected_offset * change_mask  # (B, T)
        return selected_offset  # (B, T)
        
    def make_button_hvo(self, input, latent):
        """
        Args:
            input: Tensor of shape (B, T, E, M)
            latent: Tensor of shape (B, T, M-1)
        Returns:
            button_hvo: (B, T, num_buttons, M) — [activation_flag, velocity, offset]
            change_mask: (B, T) — boolean mask of time steps where button changed
        """
        # Get hits and velocity from latent
        hits_latent = latent[:, :, 0]       # (B, T)
        velocity_latent = latent[:, :, 1]   # (B, T)

        # Quantize hits and normalize velocity
        button_hits_seq = self.quantize(hits_latent) # (B, T), values in [-1, 1]
        button_velocity_seq = torch.sigmoid(velocity_latent) # (B, T)

        
        B, T = button_hits_seq.shape
        M = input.shape[-1]
        # Fix device issue: create tensor on same device as input
        button_hvo = torch.zeros(B, T, self.num_buttons, M, device=input.device, dtype=input.dtype)

        # Identify time steps where button changes
        change_mask = torch.zeros_like(button_hits_seq, dtype=torch.bool)
        change_mask[:, 0] = True
        change_mask[:, 1:] = button_hits_seq[:, 1:] != button_hits_seq[:, :-1]  # (B, T)
        #print(f"change_mask: {change_mask[0, :]}")

        indices    = self.hit_quantizer.real_to_discrete(button_hits_seq) # (B, T)
        #print(f"indices: {indices[0, :]}")
        velocities = button_velocity_seq               # (B, T)
        offsets = self.pick_offset(input, change_mask) # (B, T)

        # Create index tensors on the same device as input
        batch_idx = torch.arange(B, device=input.device).unsqueeze(1).expand(B, T)  # (B, T)
        time_idx = torch.arange(T, device=input.device).unsqueeze(0).expand(B, T)   # (B, T)

        # Set hit = 1.0
        button_hvo[batch_idx, time_idx, indices, 0] = 1.0

        # Set velocity
        button_hvo[batch_idx, time_idx, indices, 1] = velocities.to(button_hvo.dtype)

        # Set offset
        button_hvo[batch_idx, time_idx, indices, 2] = offsets.to(button_hvo.dtype)

        #print(f"button_hvo: {button_hvo[0, :, :, :]}")

        return button_hvo, change_mask  # (B, T, num_buttons, M), (B, T)


    def decode(self, input, button_hvo, change_mask):
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
        memory_change_mask = ~change_mask # (B, T')

        decoder_out = self.decoder(
            tgt = target, 
            memory = memory,
            tgt_mask = target_causal_mask,
            memory_mask = memory_causal_mask,
            memory_key_padding_mask = memory_change_mask,
            tgt_is_causal = True,
            memory_is_causal = True
        ) # (B, T', D)

        output = self.output_projection(decoder_out) # (B, T', E*M)
        output = output.view(B, T, E, M) # (B, T', E*M) -> (B, T, E, M)

        h_logits = output[:, :, :, 0] # (B, T, E)
        hit_mask = (torch.sigmoid(h_logits) > 0.5).int()    # (B, T, E)
        v = torch.sigmoid(output[:, :, :, 1]) * hit_mask    # (B, T, E)
        o = torch.tanh(output[:, :, :, 2]) * 0.5 * hit_mask # (B, T, E)
        return h_logits, v, o

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
        button_hvo, change_mask = self.make_button_hvo(x, latent) # (B, T, num_buttons, M), (B, T)

        # ========== DECODING ==========
        h_logits, v, o = self.decode(x, button_hvo, change_mask) # (B, T, E), (B, T, E), (B, T, E)
        return h_logits, v, o, latent, {
            "button_hvo": button_hvo,
            "change_mask": change_mask,
        }
    
    def generate(self, input, button_hvo, change_mask):
        """
        Generate a prediction for the input at time t, given input < t, button HVO <= t, and change mask <= t.
        Args:
            input: Tensor of shape (B, T, E, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            change_mask: Tensor of shape (B, T)
        Returns:
            hvo_pred: Tensor of shape (B, 1, E, 3)
        """
        B, T, E, M = input.shape
        num_buttons = button_hvo.shape[2]
        target = self.dec_inp_proj(input.view(B, T, E * M)) # (B, T', D)
        target = self.pos_emb(target) # (B, T', D)
        target_causal_mask = CausalMask(target) # (T', T')
        
        memory = self.dec_button_proj(button_hvo.view(B, T, num_buttons * M))  # (B, T', D)
        memory = self.pos_emb(memory) # (B, T', D)
        memory_causal_mask = CausalMask(memory) # (T', T')
        memory_change_mask = ~change_mask # (B, T')

        decoder_out = self.decoder(
            tgt = target, 
            memory = memory,
            tgt_mask = target_causal_mask,
            memory_mask = memory_causal_mask,
            memory_key_padding_mask = memory_change_mask,
            tgt_is_causal = True,
            memory_is_causal = True
        ) # (B, T', D)

        output = self.output_projection(decoder_out) # (B, T', E*M)
        output = output.view(B, T, E, M) # (B, T', E*M) -> (B, T', E, M)

        pred = output[:, -1, :, :] # (B, E, M)

        h_logits = pred[:, :, 0] # (B, E)
        h_pred = (torch.sigmoid(h_logits) > 0.5).int()    # (B, E)
        v_pred = torch.sigmoid(pred[:, :, 1]) * h_pred    # (B, E)
        o_pred = torch.tanh(pred[:, :, 2]) * 0.5 * h_pred # (B, E)
        hvo_pred = torch.stack([h_pred, v_pred, o_pred], dim=-1) # (B, E, 3)
        hvo_pred = hvo_pred.unsqueeze(1) # (B, 1, E, 3)
        return hvo_pred



        



if __name__ == "__main__":
    input_size = (4, 33, 9, 3)
    encoder = IQAE(T=33, E=9, M=3, embed_dim=128, encoder_depth=4, encoder_heads=4, decoder_depth=1, decoder_heads=2, num_buttons=2)
    summary(encoder, input_size=input_size)


        