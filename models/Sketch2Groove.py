import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from .sub_modules import (
    DrumAxialTransformer, 
    PositionalEncoding, 
    CausalMask, 
    TransformerDecoderLayer
)

class Sketch2Groove(nn.Module):
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
            z_dim=64, supervised_dim=0,
            embed_dim=128, encoder_depth=4, encoder_heads=4,
            decoder_depth=2, decoder_heads=4, 
            num_buttons=2
    ):
        """
        Args:
            T (int): maximum length this model can generate.
            E (int): Number of drum instruments.
            M (int): Number of expressive features.
            z_dim (int): Dimension of the latent vector.
            supervised_dim (int): Number of dimensions of z_dim for supervised conditioning.
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
        self.z_dim = z_dim
        self.supervised_dim  = supervised_dim
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
        
        self.z_mu_proj         = nn.Linear(embed_dim, z_dim)
        self.z_logvar_proj     = nn.Linear(embed_dim, z_dim)
        self.attn_proj_instr   = nn.Linear(embed_dim, 1)
        self.time_attn_proj    = nn.Linear(embed_dim, 1)

        self.dec_inp_proj      = nn.Linear(E * M, embed_dim) # (B, T', E*M) -> (B, T', D)
        self.dec_button_proj   = nn.Linear(z_dim + num_buttons * M, embed_dim) # (B, T', z_dim + num_buttons*M) -> (B, T', D)
        
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
    
    def aggregate_time(self, encoded):
        # (B, T, D) -> (B, D)
        attn_weights = F.softmax(self.time_attn_proj(encoded), dim=1)
        latent = torch.sum(encoded * attn_weights, dim=1)
        return latent

    def encode(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
        Returns:
            z: Tensor of shape (B, z_dim)
            mu: Tensor of shape (B, z_dim)
            kl_loss: scalar
        """
        B, T, E, M = x.shape
        encoded = self.encoder(x)                   # (B, T, E, D)
        
        # z
        latent = self.aggregate_instrument(encoded) # (B, T, D)
        latent = self.aggregate_time(latent) # (B, D)
        mu = self.z_mu_proj(latent)          # (B, z_dim)
        logvar = self.z_logvar_proj(latent)  # (B, z_dim)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std  # Reparameterization  # (B, z_dim)
        kl_loss = (-0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)).mean() # (B)
        return z, mu, kl_loss
    
    def supervised_regularizer(self, mu, labels):
        """
        Args:
            mu: mean tensor of shape (B, z_dim)
            labels: label tensor of shape (B, supervised_dims)
            gamma_sup: scalar weight for the loss
        Returns:
            sup_loss: scalar
        """
        mu_supervised = torch.sigmoid(mu[:, :self.supervised_dim])
        sup_loss = F.mse_loss(mu_supervised, labels, reduction='mean')
        return sup_loss
    
    def decode(self, input, button_hvo, z):
        """
        Args:
            input: Tensor of shape  (B, T, E, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            z: Tensor of shape (B, z_dim)
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
        
        combined_latent = torch.cat(
            [
                z.unsqueeze(1).expand(-1, T, -1), 
                button_hvo.view(B, T, num_buttons * M)
            ], dim=2) # (B, T, z_dim + num_buttons * M)
        memory = self.dec_button_proj(combined_latent)  # (B, T', D)
        memory = self.pos_emb(memory) # (B, T', D)
        memory_causal_mask = CausalMask(memory) # (T', T')

        decoder_out = self.decoder(
            tgt = target, 
            memory = memory,
            tgt_mask = target_causal_mask,
            memory_mask = memory_causal_mask,
            tgt_is_causal = True,
            memory_is_causal = True
        ) # (B, T', D)

        output = self.output_projection(decoder_out) # (B, T', E*M)
        output = output.view(B, T, E, M) # (B, T', E*M) -> (B, T, E, M)
        h_logits = output[:, :, :, 0] # (B, T, E)
        hit_mask = (torch.sigmoid(h_logits) > 0.5).int()                 # (B, T, E)
        v = ((torch.tanh(output[:, :, :, 1]) + 1.0) / 2.0) * hit_mask    # (B, T, E)
        o = torch.tanh(output[:, :, :, 2]) * 0.5 * hit_mask # (B, T, E)

        attn_weights = {
            'self_attn': self.decoder.layers[0].self_attn_weights,   # first layer (B, T', T')
            'cross_attn': self.decoder.layers[-1].cross_attn_weights # last layer (B, T', T')
        }
        return h_logits, v, o, attn_weights
    
    def sample_z(self, batch_size, device):
        """
        Sample z from standard normal prior
        """
        return torch.randn(batch_size, self.z_dim, device=device)

    def forward(self, x, button_hvo, labels=None):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            labels: Tensor of shape (B, supervised_dim)
        Returns:
            h_logits: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
            attn_weights: Dictionary of attention weights
            kl_loss: Tensor of shape (B)
            sup_loss: Tensor of shape (B)
        """
        # ========== ENCODING ==========
        z, mu, kl_loss = self.encode(x) # (B, z_dim), (B, z_dim), (B)

        # ========== DECODING ==========
        h_logits, v, o, attn_weights = self.decode(x, button_hvo, z) # (B, T, E), (B, T, E), (B, T, E)

        sup_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            sup_loss = self.supervised_regularizer(mu, labels)

        return {
                'h_logits': h_logits, 
                'v': v, 
                'o': o, 
                'attn_weights': attn_weights, 
                'kl_loss': kl_loss, 
                'sup_loss': sup_loss
            }
    
    def generate(self, button_hvo, z=None, max_steps=None):
        """
        Generate a prediction for the input at time t, given input < t, button HVO <= t, and latent vector z.
        Args:
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            z: Tensor of shape (B, z_dim)
            max_steps: int (optional)
        Returns:
            hvo_pred: Tensor of shape (B, T, E, 3)
            hit_logits: Tensor of shape (B, T, E) for threshold calculation
        """
        if z is None:
            z = self.sample_z(button_hvo.shape[0], button_hvo.device)
        
        B, T, num_buttons, M = button_hvo.shape
        E = self.E
        T_gen = max_steps or T

        generated = self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1) # (B, 1, E, M)
        hit_probs = []
        for t in range(T_gen):

            # Target
            tgt_embed = self.dec_inp_proj(generated.view(B, t + 1, E * M)) # (B, T, D)
            tgt_embed_pos = self.pos_emb(tgt_embed)
            
            # Memory
            combined_latent = torch.cat(
                [
                    z.unsqueeze(1).expand(-1, t + 1, -1), 
                    button_hvo[:, :t + 1].view(B, t + 1, num_buttons * M)
                ], dim=2
            ) # (B, T, z_dim + num_buttons * M)
            mem_embed = self.dec_button_proj(combined_latent) # (B, T, D)
            mem_embed_pos = self.pos_emb(mem_embed)
            
            # Mask
            tgt_mask = CausalMask(tgt_embed_pos)
            mem_mask = CausalMask(mem_embed_pos)
           
            dec_out = self.decoder(
                tgt = tgt_embed_pos, 
                memory = mem_embed_pos, 
                tgt_mask = tgt_mask, 
                memory_mask = mem_mask, 
                tgt_is_causal = True,   
                memory_is_causal = True
            ) # (B, t + 1, D)

            # Output
            output = self.output_projection(dec_out) # (B, t + 1, E * M)
            output = output.view(B, t + 1, E, M)     # (B, t + 1, E, M)

            # Predict
            pred_step = output[:, -1, :, :]          # (B, E, M)
            h_logits = pred_step[:, :, 0]            # (B, E)
            h_prob = torch.sigmoid(h_logits)
            hit_probs.append(h_prob)

            # Predict
            h_pred = (h_prob > 0.5).int() # (B, E)
            v_pred = ((torch.tanh(pred_step[:, :, 1]) + 1.0) / 2.0) * h_pred 
            o_pred = torch.tanh(pred_step[:, :, 2]) * 0.5 * h_pred    
            hvo_pred  = torch.stack([h_pred, v_pred, o_pred], dim=-1) # (B, E, 3)
            generated = torch.cat([generated, hvo_pred.unsqueeze(1)], dim=1) # (B, t + 1, E, M)

        return generated, torch.stack(hit_probs, dim=1) # (B, T, E)

if __name__ == "__main__":
    input_size  = (4, 33, 9, 3)
    button_size = (4, 33, 3, 3)
    model = Sketch2Groove(T=33, E=9, M=3, 
                          z_dim=64, supervised_dim=4,
                          embed_dim=128, encoder_depth=2, encoder_heads=2, 
                          decoder_depth=1, decoder_heads=1, 
                          num_buttons=3)
    summary(model, input_size=(input_size, button_size))


        