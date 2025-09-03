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

class GrooveIQ(nn.Module):
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
            chunk_size: int = 16,
            p: float = 0.5,
            button_penalty: int = 1,
            button_dropout: float = 0.5,
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
        self.p = p
        self.button_penalty = button_penalty
        self.threshold = 0.5
        self.temperature = 1.0

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

        # Prior p(z|button) via RNN
        self.latent_prior    = nn.LSTM(embed_dim, z_dim, batch_first=True)
        self.z_prior_mu      = nn.Linear(z_dim, z_dim)
        self.z_prior_logvar  = nn.Linear(z_dim, z_dim)

        self.dec_inp_proj = nn.Linear(E * M, embed_dim) # (B, T', E*M) -> (B, T', D)
        self.dec_z_proj   = nn.Linear(z_dim, embed_dim) # (B, T', z_dim) -> (B, T', D)
        self.dec_btn_proj = nn.Linear(embed_dim, embed_dim) # (B, T', D) -> (B, T', D)
        
        self.decoder = nn.TransformerDecoder(
            decoder_layer = TransformerDecoderLayer(d_model=embed_dim, nhead=decoder_heads, batch_first=True, norm_first=True),
            num_layers    = decoder_depth,
            norm          = nn.LayerNorm(embed_dim)
        )
        
        self.output_projection = nn.Linear(embed_dim, E * M) # (B, T', D) -> (B, T', E*M)

    def chunk(self, tensor):
        """
        Args:
            tensor: Tensor of shape (B, T, D)
        Returns:
            chunked: Tensor of shape (B, T_, D)
        """
        B, T, D = tensor.shape
        T_ = T // self.chunk_size
        chunked = tensor[:, :T_ * self.chunk_size].view(B, T_, self.chunk_size, D).mean(dim=2) # (B, T_, D)
        return chunked
    
    def expand(self, tensor):
        """
        Args:
            tensor: Tensor of shape (B, T_, D)
        Returns:
            expanded: Tensor of shape (B, T_ * chunk_size, D)
        """
        B, T_, D = tensor.shape
        expanded = tensor.unsqueeze(2).repeat(1, 1, self.chunk_size, 1).view(B, T_ * self.chunk_size, D) # (B, T_ * chunk_size, D)
        return expanded # (B, T_ * chunk_size, D)
    
    def chunk_and_expand(self, tensor):
        """
        Args:
            tensor: Tensor of shape (B, T, D)
            chunk_size: int
        Returns:
            expanded: Tensor of shape (B, T_ * chunk_size, D)
        """
        return self.expand(self.chunk(tensor))

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

    def make_z_post(self, button_embed, encoded_chunked):
        """
        Args:
            button_embed: Tensor of shape (B, T, D)
            encoded_chunked: Tensor of shape (B, T // chunk_size, D)
        Returns:
            z_post: Tensor of shape (B, T, z_dim)
            mu: Tensor of shape (B, T, z_dim)
            logvar: Tensor of shape (B, T, z_dim)
        """
        btn_pooled = self.chunk_and_expand(button_embed) # (B, T_, D)
        enc_pooled = self.expand(encoded_chunked) # (B, T_, D)
        
        # Posterior Network q(z | x, button_hits)
        enc_mask_cat = torch.cat([enc_pooled, btn_pooled], dim=-1)        # (B, T, 2 * D)
        mu = self.z_mu_proj(enc_mask_cat)                                 # (B, T, z_dim)
        logvar = self.z_logvar_proj(enc_mask_cat).clamp(min=-10, max=10)  # (B, T, z_dim)
        std = torch.exp(0.5 * logvar)                                     # (B, T, z_dim)
        eps = torch.randn_like(std)                                       # (B, T, z_dim)
        z_post = mu + eps * std                                           # (B, T, z_dim)
        return z_post, mu, logvar 

    def make_z_prior(self, button_embed):
        """
        Args:
            button_embed: Tensor of shape (B, T, D)
        Returns:
            z_prior: Tensor of shape (B, T, z_dim)
            mu_prior: Tensor of shape (B, T, z_dim)
            logvar_prior: Tensor of shape (B, T, z_dim)
        """
        # Prior Network p(z_t | button_hits)
        prior_embed, _ = self.latent_prior(button_embed) # (B, T, z_dim)
        prior_embed = self.chunk_and_expand(prior_embed) # (B, T, z_dim)
        mu = self.z_prior_mu(prior_embed)                                # (B, T, z_dim)
        logvar = self.z_prior_logvar(prior_embed).clamp(min=-10, max=10) # (B, T, z_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z_prior = mu + eps * std
        return z_prior, mu, logvar
    
    def calculate_kl_loss(self, z_post, z_prior, mu_post, mu_prior, logvar_post, logvar_prior):
        """
        Args:
            z_post: Tensor of shape (B, T, z_dim)
            z_prior: Tensor of shape (B, T, z_dim)
        """
        B, T, _ = z_post.shape
        var_post = torch.exp(logvar_post)
        var_prior = torch.exp(logvar_prior)

        kl = 0.5 * torch.sum(
            logvar_prior - logvar_post + (var_post + (mu_post - mu_prior)**2) / (var_prior + 1e-6) - 1
        ) / (B * T)
        return kl

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
    
    def sample_z_from_button_embed(self, button_embed):
        """
        Args:
            button_embed: Tensor of shape (B, T, D)
        Returns:
            z: Tensor of shape (B, T, z_dim)
        """
        z_prior, mu_prior, logvar_prior = self.make_z_prior(button_embed)
        return z_prior

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
        button_embed = self.button_dropout(button_embed)   # (B, T, D)
        
        # Posterior network
        encoded_chunked = self.chunk(encoded) # (B, T // chunk_size, D)
        z_post, mu_post, logvar_post = self.make_z_post(button_embed, encoded_chunked) # (B, T, z_dim), (B, T, z_dim), (B, T, z_dim)

        # Prior network
        z_prior, mu_prior, logvar_prior = self.make_z_prior(button_embed) # (B, T, z_dim), (B, T, z_dim), (B, T, z_dim)

        # Calculate KL loss/Distill loss
        kl_loss = self.calculate_kl_loss(z_post, z_prior, mu_post, mu_prior, logvar_post, logvar_prior)
        distill_loss = F.mse_loss(mu_prior, mu_post) + F.mse_loss(logvar_prior, logvar_post)


        # Scheduled sampling from prior network: 1 → use z_post, 0 → use z_prior
        mask = torch.bernoulli(torch.full((x.shape[0], 1, 1), self.p, device=x.device))
        z = mask * z_post + (1 - mask) * z_prior 
        
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
                'kl_loss': kl_loss, 
                'distill_loss': distill_loss,
                'button_penalty': button_penalty,
            }
    
    def generate(self, button_embed, z=None, max_steps=None, threshold=None, temperature=None):
        """
        Generate a prediction for the input at time t, given input < t, button HVO <= t, and latent vector z.
        Args:
            z: Tensor of shape (B, T, z_dim)
            button_embed: Tensor of shape (B, T, D)
            max_steps: int (optional)
            threshold: float or tensor broadcastable to (B, E); compared in probability space
            temperature: float or tensor broadcastable to (B, E); applied to hit logits as p = sigmoid(logits / T)
        Returns:
            hvo_pred: Tensor of shape (B, T, E, 3)
            hit_logits: Tensor of shape (B, T, E) for threshold calculation
        """
        B, T, _ = button_embed.shape
        E = self.E
        T_gen = int(max_steps or T)

        if threshold is None:
            threshold = self.threshold

        if temperature is None:
            temperature = self.temperature

        generated = self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1) # (B, 1, E, M)
        hit_probs = []
        if z is None:
            z = self.sample_z_from_button_embed(button_embed)
        
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
            h_prob = torch.sigmoid(h_logits / torch.as_tensor(temperature, device=h_logits.device, dtype=h_logits.dtype))
            hit_probs.append(h_prob)

            # Predict
            h_pred = (h_prob > torch.as_tensor(threshold, device=h_prob.device, dtype=h_prob.dtype)).int() # (B, E)
            v_pred = ((torch.tanh(pred_step[:, :, 1]) + 1.0) / 2.0) * h_pred 
            o_pred = torch.tanh(pred_step[:, :, 2]) * 0.5 * h_pred    
            hvo_pred  = torch.stack([h_pred, v_pred, o_pred], dim=-1) # (B, E, 3)
            generated = torch.cat([generated, hvo_pred.unsqueeze(1)], dim=1) # (B, t + 1, E, M)

        return generated, torch.stack(hit_probs, dim=1) # (B, T, E)

if __name__ == "__main__":
    input_size = [(4, 64, 9, 3)]
    model = GrooveIQ(
        T=64, E=9, M=3, 
        z_dim=128, embed_dim=128, 
        encoder_depth=2, encoder_heads=4, 
        decoder_depth=2, decoder_heads=2, 
        num_buttons=2, is_causal=True,
        chunk_size=16
    )
    summary(model, input_size=input_size)
    



        