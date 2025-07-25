import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from .sub_modules import LearnableBinsQuantizer

class GrooveIQ_RNN(nn.Module):
    def __init__(self, T=33, E=9, M=3, z_dim=64, supervised_dim=0,
                 embed_dim=128, encoder_depth=1, encoder_heads=4,
                 decoder_depth=1, decoder_heads=4, num_buttons=2,
                 num_bins_velocity=8, num_bins_offset=16, monotonic_alignment='hard'):
        super().__init__()

        self.T = T
        self.E = E
        self.M = M
        self.z_dim = z_dim
        self.supervised_dim = supervised_dim
        self.embed_dim = embed_dim
        self.num_buttons = num_buttons
        self.is_velocity_quantized = num_bins_velocity > 0
        self.is_offset_quantized = num_bins_offset > 0
        self.monotonic_alignment = monotonic_alignment
        self.sos_token = nn.Parameter(torch.randn(1, E, M))
        self.pos_emb = nn.Identity()

        self.encoder = nn.GRU(input_size=M, hidden_size=embed_dim,
                              num_layers=encoder_depth, batch_first=True, bidirectional=True)
        self.encoder_out_proj = nn.Linear(embed_dim * 2, embed_dim)

        self.z_mu_proj       = nn.Linear(embed_dim, z_dim)
        self.z_logvar_proj   = nn.Linear(embed_dim, z_dim)
        self.attn_proj_instr = nn.Linear(embed_dim, 1)
        self.time_attn_proj  = nn.Linear(embed_dim, 1)
        self.button_projection = nn.Linear(embed_dim, num_buttons * M)
        self.align_proj        = nn.Linear(num_buttons * M, embed_dim)
        self.dec_inp_proj    = nn.Linear(E * M, embed_dim)
        self.dec_button_proj = nn.Linear(z_dim + num_buttons * M, embed_dim)

        self.decoder = nn.GRU(input_size=embed_dim, hidden_size=embed_dim,
                              num_layers=decoder_depth, batch_first=True)

        self.output_projection = nn.Linear(embed_dim, E * M)
        self.velocity_quantizer = LearnableBinsQuantizer(num_bins_velocity, min_val=0.0, max_val=1.0)
        self.offset_quantizer = LearnableBinsQuantizer(num_bins_offset, min_val=-0.5, max_val=0.5)

    def aggregate_instrument(self, encoded):
        """
        Args:
            encoded: Tensor of shape (B, T, embed_dim)
        Returns:
            aggregated: Tensor of shape (B, embed_dim)
        """
        attn_scores = self.attn_proj_instr(encoded).squeeze(-1) # (B, T)
        attn_weights = F.softmax(attn_scores, dim=-1) # (B, T)
        return torch.sum(encoded * attn_weights.unsqueeze(-1), dim=2) # (B, embed_dim)

    def aggregate_time(self, encoded):
        """
        Args:
            encoded: Tensor of shape (B, T, embed_dim)
        Returns:
            aggregated: Tensor of shape (B, embed_dim)
        """
        attn_weights = F.softmax(self.time_attn_proj(encoded), dim=1) # (B, T)
        return torch.sum(encoded * attn_weights, dim=1) # (B, embed_dim)

    def encode(self, x):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
        Returns:
            button_latent: Tensor of shape (B, T, num_buttons, M)
            z: Tensor of shape (B, z_dim)
            mu: Tensor of shape (B, z_dim)
            kl_loss: Tensor of shape (B)
        """
        B, T, E, M = x.shape
        x_flat = x.view(B * E, T, M)
        encoded, _ = self.encoder(x_flat)        # (B * E, T, embed_dim * 2)
        encoded = self.encoder_out_proj(encoded) # (B * E, T, embed_dim)
        encoded = encoded.view(B, E, T, self.embed_dim).permute(0, 2, 1, 3) # (B, T, E, embed_dim)
        latent = self.aggregate_instrument(encoded) # (B, T, embed_dim)

        button_latent = self.button_projection(latent).view(B, T, self.num_buttons, self.M) # (B, T, num_buttons, M)
        latent_time = self.aggregate_time(latent) # (B, embed_dim)
        mu = self.z_mu_proj(latent_time)          # (B, z_dim)
        logvar = self.z_logvar_proj(latent_time)  # (B, z_dim)
        std = torch.exp(0.5 * logvar)             # (B, z_dim)
        eps = torch.randn_like(std)               # (B, z_dim)
        z = mu + eps * std                        # (B, z_dim)
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) # (B)
        return button_latent, z, mu, kl_loss
    
    def straight_through_binarize(self, x, threshold=0.5):
        """Applies hard threshold during forward, identity gradient during backward."""
        hard = (x > threshold).float()
        return x + (hard - x).detach()
    
    def make_button_hvo(self, button_latent, button_hvo_target=None):
        """
        Args:
            latent: Tensor of shape (B, T, num_buttons, M)
            button_hvo_target: Tensor of shape (B, T, num_buttons, M) A heuristic target for the button HVO.
        Returns:
            button_hvo: (B, T, num_buttons, M) — [hit, velocity, offset]
        """
        # Get hits and velocity from latent
        hits_latent     = button_latent[:, :, :, 0]   # (B, T, num_buttons)
        velocity_latent = button_latent[:, :, :, 1]   # (B, T, num_buttons)
        offset_latent   = button_latent[:, :, :, 2]   # (B, T, num_buttons)
        hits_latent     = torch.sigmoid(hits_latent)
        velocity_latent = (torch.tanh(velocity_latent) + 1.0) / 2.0 # [-1.0, 1.0] -> [0, 1]
        offset_latent   = torch.tanh(offset_latent) * 0.5           # [-1.0, 1.0] -> [-0.5, 0.5]

        heuristic_loss = torch.tensor(0.0, device=button_latent.device)
        if button_hvo_target is not None:
            if self.monotonic_alignment == 'hard':
                heuristic_loss = self.hard_monotonic_alignment(button_hvo_target, torch.stack([hits_latent, velocity_latent, offset_latent], dim=-1))
            elif self.monotonic_alignment == 'soft':
                heuristic_loss = self.soft_monotonic_alignment(button_hvo_target, torch.stack([hits_latent, velocity_latent, offset_latent], dim=-1))

        # Quantize
        hits_latent     = self.straight_through_binarize(hits_latent)
        velocity_latent = velocity_latent if not self.is_velocity_quantized else self.velocity_quantizer(velocity_latent)
        offset_latent   = offset_latent if not self.is_offset_quantized else self.offset_quantizer(offset_latent)
        button_hvo      = torch.stack([hits_latent, velocity_latent, offset_latent], dim=-1) # (B, T, num_buttons, M)
        return button_hvo, heuristic_loss
    
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
    
    def hard_monotonic_alignment(self, button_hvo_target, button_hvo_pred):
        """
        Args:
            button_hvo_target: Tensor of shape (B, T, num_buttons, M)
            button_hvo_pred: Tensor of shape (B, T, num_buttons, M)
        Returns:
            loss: scalar
        """
        weight = torch.tensor(15.0, device=button_hvo_pred.device)
        hit_mask = (button_hvo_target[:, :, :, 0] == 1).float() # (B, T, num_buttons)
        hit_mask = hit_mask * weight
        hit_loss = F.binary_cross_entropy_with_logits(
            button_hvo_pred[:, :, :, 0], 
            button_hvo_target[:, :, :, 0], 
            reduction='mean',
            pos_weight=weight
        )
        velocity_loss = (F.mse_loss(
            button_hvo_pred[:, :, :, 1], 
            button_hvo_target[:, :, :, 1], 
            reduction='none'
        ) * hit_mask).mean()
        offset_loss = (F.mse_loss(
            button_hvo_pred[:, :, :, 2], 
            button_hvo_target[:, :, :, 2], 
            reduction='none'
        ) * hit_mask).mean()
        return hit_loss + velocity_loss + offset_loss
    
    def soft_monotonic_alignment(self, button_hvo_target, button_hvo_pred, temperature=1.0):
        """
        Args:
            button_hvo_target: Tensor (B, T, num_buttons, M)
            button_hvo_pred:   Tensor (B, T, num_buttons, M)
        Returns:
            loss: scalar
        """
        B, T, num_buttons, M = button_hvo_target.shape
        D = self.align_proj.out_features

        # Project to shared alignment space
        target_proj = self.align_proj(button_hvo_target.view(B, T, num_buttons * M))  # (B, T, D)
        pred_proj   = self.align_proj(button_hvo_pred.view(B, T, num_buttons * M))    # (B, T, D)

        # Compute attention energies: pred aligns to target (monotonic)
        energy = torch.matmul(pred_proj, target_proj.transpose(1, 2))  # (B, T, T)

        # Mask future time steps
        monotonic_mask = torch.tril(torch.ones(T, T, device=energy.device))  # (T, T)
        energy = energy.masked_fill(monotonic_mask == 0, float('-inf'))

        # Compute soft monotonic alignment
        alignment_probs = torch.softmax(energy / temperature, dim=-1)  # (B, T, T)

        # Expected target under alignment
        aligned_target = torch.matmul(alignment_probs, target_proj)  # (B, T, D)

        # Match prediction to soft-aligned target
        loss = F.mse_loss(aligned_target, pred_proj, reduction='mean')
        return loss

    def decode(self, input, button_hvo, z):
        """
        Args:
            input: Tensor of shape (B, T, E, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            z: Tensor of shape (B, z_dim)
        Returns:
        """
        B, T, E, M = input.shape
        num_buttons = self.num_buttons
        target = torch.cat([self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1), input[:, :-1]], dim=1)
        target = self.pos_emb(self.dec_inp_proj(target.view(B, T, E * M)))

        # Ignore info at non-hit frames
        button_hit_mask = (~(button_hvo[:, :, :, 0].sum(dim=-1) == 0)).float().unsqueeze(-1) # (B, T, 1)
        combined_latent = torch.cat([
            z.unsqueeze(1).expand(-1, T, -1),
            button_hvo.view(B, T, num_buttons * M) * button_hit_mask
        ], dim=2) # (B, T, z_dim + num_buttons * M)
        memory = self.pos_emb(self.dec_button_proj(combined_latent)) # (B, T, embed_dim)
        decoder_inp = target + memory              # (B, T, embed_dim)
        decoder_out, _ = self.decoder(decoder_inp) # (B, T, embed_dim)

        output = self.output_projection(decoder_out).view(B, T, E, M)
        h_logits = output[:, :, :, 0]
        v = (torch.tanh(output[:, :, :, 1]) + 1.0) / 2.0
        o = torch.tanh(output[:, :, :, 2]) * 0.5
        return h_logits, v, o, {}

    def sample_z(self, batch_size, device):
        return torch.randn(batch_size, self.z_dim, device=device)
    
    def forward(self, x, labels=None, button_hvo_target=None):
        """
        Args:
            x: Tensor of shape (B, T, E, M)
            labels: Tensor of shape (B, supervised_dim)
            button_hvo_target: Tensor of shape (B, T, num_buttons, M) A heuristic target for the button HVO.
        Returns:
            h_logits: Tensor of shape (B, T, E)
            v: Tensor of shape (B, T, E)
            o: Tensor of shape (B, T, E)
            button_latent: Tensor of shape (B, T, num_button, M)
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            attn_weights: Dictionary of attention weights
            vo_penalty: Tensor of shape (B)
            kl_loss: Tensor of shape (B)
            sup_loss: Tensor of shape (B)
        """
        # ========== ENCODING ==========
        button_latent, z, mu, kl_loss = self.encode(x) # (B, T, num_buttons, M), (B, z_dim), (B, z_dim), (B)

        # ========== MAKE BUTTON HVO ==========
        button_hvo, heuristic_loss = self.make_button_hvo(button_latent, button_hvo_target) # (B, T, num_buttons, M)

        # ========== DECODING ==========
        h_logits, v, o, attn_weights = self.decode(x, button_hvo, z) # (B, T, E), (B, T, E), (B, T, E)

        button_hits     = button_hvo[:, :, :, 0] # (B, T, num_buttons)
        button_velocity = button_hvo[:, :, :, 1] # (B, T, num_buttons)
        button_offset   = button_hvo[:, :, :, 2] # (B, T, num_buttons)
        
        # Penalize offset/velocity values in non-hit frames (maybe not needed, since we mask out non-hit frames in decoder)
        no_hit_mask = (button_hits == 0).float().unsqueeze(-1) # (B, T, num_buttons, 1)
        vo_penalty  = torch.stack([
            button_velocity, 
            button_offset
        ], dim=-1) * no_hit_mask # (B, T, num_buttons, 2)
        vo_penalty = vo_penalty.abs().mean()

        sup_loss = torch.tensor(0.0, device=x.device)
        if labels is not None:
            sup_loss = self.supervised_regularizer(mu, labels)

        return {
                'h_logits': h_logits, 
                'v': v, 
                'o': o, 
                'button_latent': button_latent, 
                'button_hvo': button_hvo, 
                'attn_weights': attn_weights, 
                'vo_penalty': vo_penalty, 
                'heuristic_loss': heuristic_loss,
                'kl_loss': kl_loss, 
                'sup_loss': sup_loss
            }

    def generate(self, button_hvo, z=None, max_steps=None, threshold=0.5):
        """
        Args:
            button_hvo: Tensor of shape (B, T, num_buttons, M)
            z: Tensor of shape (B, z_dim)
            max_steps: int
            threshold: float
        Returns:
            generated: Tensor of shape (B, T_gen, E, M)
            hit_probs: Tensor of shape (B, T_gen)
        """
        if z is None:
            z = self.sample_z(button_hvo.shape[0], button_hvo.device)

        B, T, num_buttons, M = button_hvo.shape
        E = self.E
        T_gen = max_steps or T

        generated = self.sos_token.unsqueeze(0).repeat(B, 1, 1, 1)
        hidden = None
        hit_probs = []

        # Ignore info at non-hit frames
        button_hit_mask = (~(button_hvo[:, :, :, 0].sum(dim=-1) == 0)).float().unsqueeze(-1) # (B, T, 1)
        combined_latent = torch.cat([
            z.unsqueeze(1).expand(-1, T_gen, -1),
            button_hvo[:, :T_gen].view(B, T_gen, num_buttons * M) * button_hit_mask
        ], dim=2) # (B, T_gen, z_dim + num_buttons * M)
        memory_embed = self.pos_emb(self.dec_button_proj(combined_latent)) # (B, T_gen, embed_dim)

        for t in range(T_gen):
            tgt_t = generated[:, -1, :, :].view(B, E * M)
            tgt_embed = self.pos_emb(self.dec_inp_proj(tgt_t)).unsqueeze(1)
            mem_t = memory_embed[:, t:t + 1, :]
            decoder_in = tgt_embed + mem_t
            out_t, hidden = self.decoder(decoder_in, hidden)
            proj_out = self.output_projection(out_t).view(B, E, M)

            h_logits = proj_out[:, :, 0]
            h_prob = torch.sigmoid(h_logits)
            hit_probs.append(h_prob)

            h_pred = (h_prob > threshold).float()
            v_pred = ((torch.tanh(proj_out[:, :, 1]) + 1) / 2) * h_pred
            o_pred = torch.tanh(proj_out[:, :, 2]) * 0.5 * h_pred

            step_out = torch.stack([h_pred, v_pred, o_pred], dim=-1)
            generated = torch.cat([generated, step_out.unsqueeze(1)], dim=1)
        return generated, torch.stack(hit_probs, dim=1)
    
if __name__ == "__main__":
    input_size = (4, 33, 9, 3)
    encoder = GrooveIQ_RNN(
        T=33, E=9, M=3, z_dim=64, supervised_dim=4,
        embed_dim=128, encoder_depth=4, encoder_heads=4, 
        decoder_depth=1, decoder_heads=2, 
        num_buttons=3, num_bins_velocity=8, num_bins_offset=16,
        monotonic_alignment='hard'
    )
    summary(encoder, input_size=input_size)
