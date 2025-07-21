from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from models import Sketch2Groove
import torchaudio.functional as aF
from data import DrumMIDIFeature
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import precision_recall_curve
import numpy as np
import seaborn as sns
import os
import wandb

class Sketch2Groove_Trainer(BaseTrainer):
    """
    Sketch2Groove (Sketch to Groove) Trainer class that handles training, validation, and recognition loops.
    """
    def __init__(self, model, config, run_name, config_file, device=None):
        super().__init__(model, config, run_name, config_file, device)
        
        #  Loss weights
        self.pos_weight  = config['loss'].get('pos_weight', 1.0)
        self.hit_penalty = config['loss'].get('hit_penalty', 1.0)
        self.threshold   = config['loss'].get('threshold', 0.5)
        self.recons_weight = config['loss'].get('recons_weight', 1.0)
        self.kld_weight    = config['loss'].get('kld_weight', 1.0)
        self.sup_weight        = config['loss'].get('sup_weight', 1.0)
        
        # Reconstruction losses
        self.hit_loss      = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(self.pos_weight))
        self.velocity_loss = nn.MSELoss(reduction='none')
        self.offset_loss   = nn.MSELoss(reduction='none')


    def set_optimizer(self, optimizer) -> None:
        self.optimizer = optimizer

    def set_scheduler(self, scheduler) -> None:
        self.scheduler = scheduler

    def _train_epoch(self, dataloader):
        """
        Train for one epoch.
        Args:
            dataloader: DataLoader for training data
        Returns:
            Tuple[Dict[str, float], Dict[str, Any]]: Average metrics and data to plot
        """
        self.model.train()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Training GrooveIQ]")

        # Initialize accumulators
        running_kld_loss = 0.0
        running_sup_loss = 0.0
        running_joint_loss = 0.0
        running_sample_count = 0
        

        # hit/velocity/offset metrics
        running_hit_bce = 0.0
        running_hit_acc = 0.0
        running_hit_ppv = 0.0
        running_hit_tpr = 0.0
        running_hit_f1  = 0.0
        running_hit_perplexity = 0.0
        running_velocity_mse   = 0.0
        running_offset_mse     = 0.0

        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            grids, button_hvo, samples, labels = batch['grid'].to(self.device), batch['button_hvo'].to(self.device), batch['samples'], batch['labels']
            if labels is not None:
                labels = labels.float().to(self.device)
            h_true, v_true, o_true = grids[:, :, :, 0], grids[:, :, :, 1], grids[:, :, :, 2]

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                outputs  = self.model(grids, button_hvo, labels)
                h_logits = outputs['h_logits']
                h_mask   = (torch.sigmoid(h_logits) > self.threshold).int()
                v_pred   = outputs['v'] * h_mask
                o_pred   = outputs['o'] * h_mask
                attn_weights  = outputs['attn_weights']
                kl_loss  = outputs['kl_loss']
                sup_loss = outputs['sup_loss']

                # Hit penalty for penalizing velocity/offset when there is no hit
                hit_penalty = torch.where(h_true == 1, self.hit_penalty, 0.0)

                # Reconstruction lossses
                hit_bce = self.hit_loss(h_logits, h_true).mean()
                velocity_mse = (self.velocity_loss(v_pred, v_true) * hit_penalty).mean()
                offset_mse = (self.offset_loss(o_pred, o_true) * hit_penalty).mean()

                # Joint loss
                joint_loss = self.recons_weight * (hit_bce + velocity_mse + offset_mse) + \
                             self.kld_weight * kl_loss + \
                             self.sup_weight * sup_loss

            # Compute hit metrics
            hit_pred_int = (torch.sigmoid(h_logits) > self.threshold).int()
            h_true_int = h_true.int()
            hit_tp = ((hit_pred_int == 1) & (h_true_int == 1)).sum().item()
            hit_fp = ((hit_pred_int == 1) & (h_true_int == 0)).sum().item()
            hit_fn = ((hit_pred_int == 0) & (h_true_int == 1)).sum().item()
            hit_tn = ((hit_pred_int == 0) & (h_true_int == 0)).sum().item()
            hit_acc = (hit_tp + hit_tn) / (hit_tp + hit_fp + hit_fn + hit_tn) if (hit_tp + hit_fp + hit_fn + hit_tn) > 0 else 0.0
            hit_ppv = hit_tp / (hit_tp + hit_fp) if (hit_tp + hit_fp) > 0 else 0.0
            hit_tpr = hit_tp / (hit_tp + hit_fn) if (hit_tp + hit_fn) > 0 else 0.0
            hit_f1 = (2 * hit_tp) / (2 * hit_tp + hit_fp + hit_fn) if (2 * hit_tp + hit_fp + hit_fn) > 0 else 0.0
            hit_perplexity = torch.exp(hit_bce).item()

            # Accumulate metrics
            batch_size = grids.size(0)
            running_sample_count += batch_size
            running_hit_bce += hit_bce.item() * batch_size
            running_velocity_mse += velocity_mse.item() * batch_size
            running_offset_mse += offset_mse.item() * batch_size
            running_kld_loss += kl_loss.item() * batch_size
            running_sup_loss += sup_loss.item() * batch_size
            running_joint_loss += joint_loss.item() * batch_size
            running_hit_acc += hit_acc * batch_size
            running_hit_ppv += hit_ppv * batch_size
            running_hit_tpr += hit_tpr * batch_size
            running_hit_f1 += hit_f1 * batch_size
            running_hit_perplexity += hit_perplexity * batch_size

            # Gradient accumulation
            scaled_loss = joint_loss / self.config['training']['gradient_accumulation_steps']
            self.scaler.scale(scaled_loss).backward()

            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            # Update progress bar
            avg_hit_bce = running_hit_bce / running_sample_count
            avg_hit_perplexity = running_hit_perplexity / running_sample_count
            avg_velocity_mse = running_velocity_mse / running_sample_count
            avg_offset_mse = running_offset_mse / running_sample_count
            avg_kld_loss = running_kld_loss / running_sample_count
            avg_sup_loss = running_sup_loss / running_sample_count
            avg_joint_loss = running_joint_loss / running_sample_count
            avg_hit_acc = running_hit_acc / running_sample_count
            avg_hit_ppv = running_hit_ppv / running_sample_count
            avg_hit_tpr = running_hit_tpr / running_sample_count
            avg_hit_f1 = running_hit_f1 / running_sample_count

            batch_bar.set_postfix(
                h_bce=f"{avg_hit_bce:.4f}",
                h_acc=f"{avg_hit_acc:.4f}",
                h_ppv=f"{avg_hit_ppv:.4f}",
                h_tpr=f"{avg_hit_tpr:.4f}",
                h_f1=f"{avg_hit_f1:.4f}",
                h_perplexity=f"{avg_hit_perplexity:.4f}",
                v_mse=f"{avg_velocity_mse:.4f}",
                o_mse=f"{avg_offset_mse:.4f}",
                kld_loss=f"{avg_kld_loss:.4f}",
                sup_loss=f"{avg_sup_loss:.4f}",
                joint=f"{avg_joint_loss:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Cleanup
            torch.cuda.empty_cache()

        # Handle remaining gradients
        if (len(dataloader) % self.config['training']['gradient_accumulation_steps']) != 0:
            self.scaler.step(self.optimizer)
            if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step()
            self.scaler.update()
            self.optimizer.zero_grad()

        batch_bar.close()

        # Final metrics
        avg_metrics = {
            'hit_acc': running_hit_acc / running_sample_count,
            'hit_ppv': running_hit_ppv / running_sample_count,
            'hit_tpr': running_hit_tpr / running_sample_count,
            'hit_f1': running_hit_f1 / running_sample_count,
            'hit_perplexity': running_hit_perplexity / running_sample_count,
            'hit_bce': running_hit_bce / running_sample_count,
            'velocity_mse': running_velocity_mse / running_sample_count,
            'offset_mse': running_offset_mse / running_sample_count,
            'kld_loss': running_kld_loss / running_sample_count,
            'sup_loss': running_sup_loss / running_sample_count,
            'joint_loss': running_joint_loss / running_sample_count,
        }

        # Plotting
        to_plots = {
            'samples': samples,          # List of SampleData objects
            'button_hvo': button_hvo,    # Button HVO corresponding to samples  (B, T, num_buttons, M)
            'attn_weights': attn_weights # Attention weights (B, T', T')
        }

        return avg_metrics, to_plots


    def _validate_epoch(self, dataloader, num_batches: Optional[int] = None):
        """
        Validate for one epoch.
        
        Args:
            dataloader: DataLoader for validation data
        Returns:
            Tuple[Dict[str, float], List[Dict[str, Any]]]: Validation metrics and recognition results
        """
        # Generate
        results = self.generate(dataloader, num_batches=num_batches)
        
        # Extract references and hypotheses from results
        references = [result['target_grid'] for result in results]
        hypotheses = [result['generated_grid'] for result in results]
        hit_probs  = [result['hit_probs'] for result in results]
        
        # Calculate metrics on num_batches
        metrics = self._calculate_metrics(references, hypotheses, hit_probs)
        return metrics, results
    
    def train(self, train_dataloader, val_dataloader, epochs: Optional[int] = None):
        """
        Full training loop for IQAE training.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Optional[int], number of epochs to train
        """
        super().train(train_dataloader, val_dataloader)

        # Training loop
        best_joint_loss = float('inf')
        best_hit_acc    = 0

        if epochs is None:
            epochs = self.config['training']['epochs']

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            
            self.current_epoch += 1
            train_metrics, train_plots = self._train_epoch(train_dataloader)
            val_metrics, val_results = self._validate_epoch(val_dataloader, num_batches=10)
            self.threshold = val_metrics['optimal_threshold']

            # Step ReduceLROnPlateau scheduler with validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['hit_acc'])
            
            # Log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # Save plots / midi / checkpoints
            self._save_midi_plots(train_plots, val_results, epoch)
            self._save_attention_plot(train_plots['attn_weights']['self_attn'][0], epoch, attn_type="self")
            self._save_attention_plot(train_plots['attn_weights']['cross_attn'][0], epoch, attn_type="cross")
            self._save_midi(val_results, epoch)
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            # Check if this is the best model
            if val_metrics['hit_acc'] > best_hit_acc:
                best_hit_acc = val_metrics['hit_acc']
                self.best_metric = val_metrics['hit_acc']
                self.save_checkpoint('checkpoint-best-metric-model.pth') 

    def evaluate(self, test_dataloader, num_batches: Optional[int] = None):
        """
        Evaluate the model on the test set.
        """
        raise NotImplementedError("Evaluation is not implemented for GrooveIQ")
                

    def generate(self, dataloader, max_length: Optional[int] = None, num_batches: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Generate grids from button HVO.
        Args:
            dataloader: DataLoader containing the evaluation data
            max_length: Optional[int], maximum length of the generated sequence
            num_batches: Optional[int], number of batches to generate
        Returns:
            List of dictionaries containing recognition results with generated sequences and scores
        """
        # Set max length
        if max_length is None:
            max_length = self.model.T

        # Initialize variables
        self.model.eval()
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc=f"[Generating]")
        results = []

        # Run inference
        with torch.inference_mode():
            for i, batch in enumerate(dataloader):
                
                grids, button_hvo, samples = batch['grid'].to(self.device), batch['button_hvo'].to(self.device), batch['samples']
                z, _, _ = self.model.encode(grids)
                generated_grids, hit_probs = self.model.generate(button_hvo, z, max_steps=max_length, threshold=self.threshold)

                torch.cuda.empty_cache()

                # Post process sequences: 
                # 1. Convert to Sample objects
                # 2. Calculate some metrics
                num_samples = generated_grids.shape[0] # (B)
                for b in range(num_samples):
                    target_sample = samples[b]
                    target_grid = grids[b, :, :, :] # (T, E, 3)
                    generated_grid = generated_grids[b, 1:, :, :] # Drop SOS token, (T', E, 3)
                    try:
                        generated_sample = samples[b].from_fixed_grid(generated_grid, steps_per_quarter=self.config['data']['steps_per_quarter'])
                    except Exception as e:
                        print(f"Error generating sample: {e}")
                        generated_sample = None
                    
                    results.append({
                        'generated_sample': generated_sample,
                        'generated_grid': generated_grid,
                        'target_sample': target_sample,
                        'target_grid': target_grid,
                        'hit_probs': hit_probs[b, :, :].cpu().detach(),
                        'button_hvo': button_hvo[b, :, :, :].cpu().detach()
                    })
                
                # Update progress bar
                batch_bar.update()
                if num_batches is not None and i >= num_batches - 1:
                    break

            batch_bar.close()
            return results

        
    def _calculate_metrics(self, references: List[torch.tensor], hypotheses: List[torch.tensor], hit_probs: List[torch.tensor]) -> Dict[str, float]:
        """
        Calculate metrics for grids.
        
        Args:
            references: List of reference grid(s) where each grid is of shape (T, E, 3)
            hypotheses: List of hypothesis grid(s) where each grid is of shape (T', E, 3)
            hit_probs: List of hit probabilities where each tensor is of shape (T', E)
        Returns:
            Dictionary of metrics
        """
        hit_acc = 0.0
        hit_ppv = 0.0
        hit_tpr = 0.0
        hit_f1 = 0.0
        velocity_mse = 0.0
        offset_mse = 0.0

        # Store all hit probs and ground truths
        all_probs = []
        all_labels = []

        for reference, hypothesis, hit_prob in zip(references, hypotheses, hit_probs):

            hit_pred_int = ((hit_prob > self.threshold).int()).detach().cpu()
            h_true_int   = reference[:, :, 0].int().detach().cpu()

            hit_tp = ((hit_pred_int == 1) & (h_true_int == 1)).sum().item() # True positives
            hit_fp = ((hit_pred_int == 1) & (h_true_int == 0)).sum().item()
            hit_fn = ((hit_pred_int == 0) & (h_true_int == 1)).sum().item()
            hit_tn = ((hit_pred_int == 0) & (h_true_int == 0)).sum().item()

            total = hit_tp + hit_fp + hit_fn + hit_tn
            hit_acc += (hit_tp + hit_tn) / total if total > 0 else 0.0
            hit_ppv += hit_tp / (hit_tp + hit_fp) if (hit_tp + hit_fp) > 0 else 0.0
            hit_tpr += hit_tp / (hit_tp + hit_fn) if (hit_tp + hit_fn) > 0 else 0.0
            hit_f1  += (2 * hit_tp) / (2 * hit_tp + hit_fp + hit_fn) if (2 * hit_tp + hit_fp + hit_fn) > 0 else 0.0

            all_probs.append(hit_prob.flatten().cpu().numpy())
            all_labels.append(h_true_int.flatten().cpu().numpy())
            
            velocity_mse += F.mse_loss(reference[:, :, 1], hypothesis[:, :, 1]).item()
            offset_mse += F.mse_loss(reference[:, :, 2], hypothesis[:, :, 2]).item()

        hit_acc /= len(references)
        hit_ppv /= len(references)
        hit_tpr /= len(references)
        hit_f1  /= len(references)
        velocity_mse /= len(references)
        offset_mse /= len(references)

        # Flatten all probs and labels
        all_probs = np.concatenate(all_probs)
        all_labels = np.concatenate(all_labels)

        # Compute PR curve
        precision, recall, thresholds = precision_recall_curve(all_labels, all_probs)

        # Compute F1 at each threshold
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5  # safety fallback

        # Additional optional: best overall F1 at best threshold
        best_f1 = f1_scores[best_idx]
        best_ppv = precision[best_idx]
        best_tpr = recall[best_idx]

        return {
            'hit_acc': hit_acc,
            'hit_ppv': hit_ppv,
            'hit_tpr': hit_tpr,
            'hit_f1': hit_f1,
            'velocity_mse': velocity_mse,
            'offset_mse': offset_mse,
            'optimal_threshold': float(best_threshold),
            'optimal_f1': float(best_f1),
            'optimal_ppv': float(best_ppv),
            'optimal_tpr': float(best_tpr),
        }
    
    def _save_attention_plot(self, attn_weights: torch.Tensor, epoch: int, attn_type: str = "self"):
        """Save attention weights visualization."""
        if isinstance(attn_weights, torch.Tensor):
            attn_weights = attn_weights.cpu().detach().numpy()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_weights, cmap="viridis", cbar=True)
        plt.title(f"Attention Weights - Epoch {epoch}")
        plt.xlabel("Source Sequence")
        plt.ylabel("Target Sequence")
        
        plot_path = os.path.join(self.attn_dir, f"{attn_type}_attention_epoch{epoch}.png")
        plt.savefig(plot_path)
        plt.close()
        
        if self.use_wandb:
            wandb.log({f"{attn_type}_attention": wandb.Image(plot_path)}, step=epoch)
    
    def _save_midi_plots(self, train_plots, val_results, epoch, num_samples: int = 3):
        """
        Save plots for training and validation
        Args:
            train_plots: Dictionary containing training plots
                - samples: List of SampleData objects
                - button_hvo: Button HVO corresponding to samples (B, T, num_buttons, M)
            val_results: Dictionary containing validation results
                - generated_sample: List of SampleData objects
                - generated_grid: (B, T, E, M)
                - target_sample: List of SampleData objects
                - target_grid: (B, T, E, M)
            epoch: Current epoch
        """
        # Create plots directory if it doesn't exist
        plots_dir = self.expt_root / 'plots'
        plots_dir.mkdir(exist_ok=True)

        # TODO: Plot Training Grid Plot + Button HVO
        fig1, axes1 = plt.subplots(min(num_samples, len(train_plots['samples'])), 2, figsize=(15, 5*min(num_samples, len(train_plots['samples']))))
        fig2, axes2 = plt.subplots(min(num_samples, len(val_results)), 2, figsize=(15, 5*min(num_samples, len(val_results))))

        for i in range(min(num_samples, len(train_plots['samples']))):
            # Training grid plot
            train_plots['samples'][i].feature.fixed_grid_plot(
                ax=axes1[i, 0],
                steps_per_quarter=self.config['data']['steps_per_quarter']
            )

            # Button HVO
            DrumMIDIFeature._grid_plot(
                train_plots['button_hvo'][i, :, :, :].cpu().detach(),
                ax=axes1[i, 1],
                title="Button HVO",
                xlabel="Time Step",
                ylabel="Drum Class"
            )

        fig1.tight_layout()
        fig1.savefig(plots_dir / f"train_plots_{epoch}.png")
        plt.close(fig1)
        
        
        # Plot Generated Grid + Target Grid
        for i in range(min(num_samples, len(val_results))):
            target_sample = val_results[i]['target_sample']
            generated_sample = val_results[i]['generated_sample']
            target_sample.feature.fixed_grid_plot(
                ax=axes2[i, 0],
                steps_per_quarter=self.config['data']['steps_per_quarter']
            )

            if generated_sample is not None:  
                generated_sample.feature.fixed_grid_plot(
                    ax=axes2[i, 1],
                    steps_per_quarter=self.config['data']['steps_per_quarter']
                )
            else:
                axes2[i, 1].text(0.5, 0.5, "Failed to generate", ha='center', va='center', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(plots_dir / f"val_plots_{epoch}.png")
        plt.close(fig2)
        
    
    def _save_midi(self, val_results, epoch):
        """
        Save midi for validation
        Args:
            val_results: Dictionary containing validation results
                - generated_sample: List of SampleData objects
                - generated_grid: (B, T, E, M)
                - target_sample: List of SampleData objects
                - target_grid: (B, T, E, M)
        """
        # Create midi directory if it doesn't exist
        midi_dir = self.expt_root / 'midi'
        midi_dir.mkdir(exist_ok=True)

        results = []

        # Save generated midi
        for result in val_results:
            results.append({
                'generated_sample': result['generated_sample'].to_dict(),
                'target_sample': result['target_sample'].to_dict(),
                'button_hvo': result['button_hvo'].cpu().detach(),
            })
        
        with open(midi_dir / f"val_results_{epoch}.pkl", "wb") as f:
            pickle.dump(results, f)
