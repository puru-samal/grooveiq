from .base_trainer import BaseTrainer
from typing import Dict, Any, Optional, List, Tuple, Union
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from utils import create_scheduler, create_optimizer, ConstraintLosses
from models import IQAE
import torchaudio.functional as aF
from data import DrumMIDIFeature
import matplotlib.pyplot as plt

class IQAE_Trainer(BaseTrainer):
    """
    ASR (Automatic Speech Recognition) Trainer class that handles training, validation, and recognition loops.

    This trainer implements:
    1. Training loop with gradient accumulation, mixed precision training, and optional CTC loss
    2. Validation loop for model evaluation
    3. Recognition capabilities with different decoding strategies (greedy, beam search)
    4. Language model shallow fusion during recognition

    Implementation Tasks:
    - TODO: Implement key parts of the training loop in _train_epoch
    - TODO: Implement key parts of the validation loop in _validate_epoch
    - TODO: Calculate ASR metrics in _calculate_asr_metrics

    Implementation Notes:
    1. For __init__:
        - Initialize Hit loss
        - Initialize Velocity loss
        - Initialize Offset loss
        
    2. For _train_epoch:
        - Unpack the batch (features, shifted targets, golden targets, lengths)
        - Get model predictions, attention weights and CTC inputs
        - Calculate CE loss and CTC loss if enabled
        - Handle gradient accumulation correctly
        
    3. For _validate_epoch:
        - Use recognize() to generate transcriptions
        - Calculate WER, CER and word distance metrics
        
    4. For train:
        - Initialize scheduler if not already done
        - Set maximum transcript length
        - Implement epoch loop with training and validation
        - Handle model checkpointing and metric logging
        
    5. For recognize:
        - Initialize sequence generator with appropriate scoring function
        - Handle both greedy and beam search decoding
        - Support language model shallow fusion
        - Post-process sequences using tokenizer
    """
    def __init__(self, model, config, run_name, config_file, device=None):
        super().__init__(model, config, run_name, config_file, device)
        
        #  Initialize Hit loss
        self.pos_weight = config['loss'].get('pos_weight', 17.0)
        self.hit_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(self.pos_weight))

        #  Initialize Velocity loss
        self.velocity_loss = nn.MSELoss(reduction='none')

        #  Initialize Offset loss
        self.offset_loss = nn.MSELoss(reduction='none')

        # Initialize ConstraintLosses
        #self.temporal_loss = lambda x: ConstraintLosses().l2_temporal_diff(x)
        #self.margin_loss   = lambda x: ConstraintLosses().margin_loss(x)
        self.latent_loss   = lambda x: ConstraintLosses().l1_sparsity_time(x)

        # Hit penalty
        self.hit_penalty = config['loss'].get('hit_penalty', 1.0)

        # Kld weight
        self.kld_weight = config['loss'].get('kld_weight', 0.01)

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
        batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc="[Training IQAE]")

        # Initialize accumulators
        running_latent_penalty = 0.0
        running_kld_loss = 0.0
        running_vo_penalty = 0.0
        running_joint_loss = 0.0
        running_sample_count = 0
        

        # hit metrics
        running_hit_bce = 0.0
        running_hit_acc = 0.0
        running_hit_ppv = 0.0
        running_hit_tpr = 0.0
        running_hit_f1  = 0.0
        running_hit_perplexity = 0.0

        # velocity metrics
        running_velocity_mse = 0.0

        # offset metrics
        running_offset_mse = 0.0

        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            grids, samples = batch['grid'].to(self.device), batch['samples']
            h_true, v_true, o_true = grids[:, :, :, 0], grids[:, :, :, 1], grids[:, :, :, 2]

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                h_logits, v_pred, o_pred, button_latent, button_hvo, vo_penalty, z, kl_loss = self.model(grids)

                hit_penalty = torch.where(h_true == 1, 10.0, 0.0)

                # Hit loss
                hit_bce = self.hit_loss(h_logits, h_true).mean()

                # Velocity loss
                velocity_mse = (self.velocity_loss(v_pred, v_true) * hit_penalty).mean()

                # Offset loss
                offset_mse = (self.offset_loss(o_pred, o_true) * hit_penalty).mean()

                # Constraint on button latent
                latent_penalty = self.latent_loss(button_latent)

                # Kld loss
                kld_loss = (kl_loss * self.kld_weight)

                # Joint loss
                joint_loss = hit_bce + velocity_mse + offset_mse + vo_penalty + latent_penalty + kld_loss

            # Compute hit accuracy safely
            hit_pred_int = (torch.sigmoid(h_logits) > 0.5).int()
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
            running_vo_penalty += vo_penalty.item() * batch_size
            running_latent_penalty += latent_penalty.item() * batch_size
            running_kld_loss += kld_loss.item() * batch_size
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
            avg_latent_penalty = running_latent_penalty / running_sample_count
            avg_kld_loss = running_kld_loss / running_sample_count
            avg_vo_penalty = running_vo_penalty / running_sample_count
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
                vo_penalty=f"{avg_vo_penalty:.4f}",
                latent_penalty=f"{avg_latent_penalty:.4f}",
                kld_loss=f"{avg_kld_loss:.4f}",
                joint=f"{avg_joint_loss:.4f}",
                acc_step=f"{(i % self.config['training']['gradient_accumulation_steps']) + 1}/{self.config['training']['gradient_accumulation_steps']}"
            )
            batch_bar.update()

            # Cleanup
            #del grids
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
            'latent_penalty': running_latent_penalty / running_sample_count,
            'kld_loss': running_kld_loss / running_sample_count,
            'vo_penalty': running_vo_penalty / running_sample_count,
            'joint_loss': running_joint_loss / running_sample_count,
        }

        # Plotting
        to_plots = {
            'samples': samples,         # List of SampleData objects
            'button_hvo': button_hvo,   # Button HVO corresponding to samples  (B, T, num_buttons, M)
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
        # TODO: Recognize
        results = self.generate(dataloader, num_batches=num_batches)
        
        # TODO: Extract references and hypotheses from results
        references = [result['target_grid'] for result in results]
        hypotheses = [result['generated_grid'] for result in results]
        
        # TODO: Calculate metrics on full batch
        metrics = self._calculate_metrics(references, hypotheses)
        
        return metrics, results
    
    def train(self, train_dataloader, val_dataloader, epochs: Optional[int] = None):
        """
        Full training loop for IQAE training.
        
        Args:
            train_dataloader: DataLoader for training data
            val_dataloader: DataLoader for validation data
            epochs: Optional[int], number of epochs to train
        """
        # Some error handling
        super().train(train_dataloader, val_dataloader)

        # Training loop
        best_joint_loss = float('inf')
        best_hit_acc    = 0.0

        if epochs is None:
            epochs = self.config['training']['epochs']

        for epoch in range(self.current_epoch, self.current_epoch + epochs):
            
            self.current_epoch += 1

            # Train for one epoch
            train_metrics, train_plots = self._train_epoch(train_dataloader)
            
            # Validate
            val_metrics, val_results = self._validate_epoch(val_dataloader, num_batches=1)

            # Step ReduceLROnPlateau scheduler with validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['hit_acc'])
            
            # Log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # TODO: Generate and save plots
            self._save_plots(train_plots, val_results, epoch)
            
            # TODO: Save generated midi
            #self._save_midi(val_results, epoch)

            # Save checkpoints
            self.save_checkpoint('checkpoint-last-epoch-model.pth')
            
            # Check if this is the best model
            if val_metrics['hit_acc'] > best_hit_acc:
                best_hit_acc = val_metrics['hit_acc']
                self.best_metric = val_metrics['hit_acc']
                self.save_checkpoint('checkpoint-best-metric-model.pth') 
                

    def evaluate(self, dataloader) -> Dict[str, Dict[str, float]]:
        """
        Evaluate the model on the test set.
        
        Args:
            dataloader: DataLoader for test data
        Returns:
            Dictionary containing evaluation metrics and generated results
        """
        raise NotImplementedError("Not implemented")

    def generate(self, dataloader, max_length: Optional[int] = None, num_batches: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Evaluate the model by generating transcriptions from audio features.
        
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
                # Get data
                grids, samples = batch['grid'].to(self.device), batch['samples']
                
                # Get latent
                button_latent, z, _ = self.model.encode(grids)

                # Get button HVO
                button_hvo = self.model.make_button_hvo(button_latent)

                # Generate
                generated_grids = self.model.generate(button_hvo, z, max_steps=max_length)

                # Clean up
                #del grids, latent, button_hvo
                torch.cuda.empty_cache()

                # TODO: Post process sequences
                # Convert to Sample objects and calculate some metrics
                num_samples = generated_grids.shape[0] # (B)
                for b in range(num_samples):
                    target_sample = samples[b]
                    target_grid = grids[b, :, :, :] # (T, E, 3)
                    generated_grid = generated_grids[b, 1:, :, :] # Drop SOS token, (T', E, 3)
                    generated_sample = samples[b].from_fixed_grid(generated_grid, steps_per_quarter=self.config['data']['steps_per_quarter'])
                    
                    results.append({
                        'generated_sample': generated_sample,
                        'generated_grid': generated_grid,
                        'target_sample': target_sample,
                        'target_grid': target_grid,
                    })
                
                # Update progress bar
                batch_bar.update()
                if num_batches is not None and i >= num_batches - 1:
                    break

            batch_bar.close()
            return results

        
    def _calculate_metrics(self, references: List[torch.tensor], hypotheses: List[torch.tensor]) -> Dict[str, float]:
        """
        Calculate metrics for grids.
        
        Args:
            references: List of reference grid(s) where each grid is of shape (T, E, 3)
            hypotheses: List of hypothesis grid(s) where each grid is of shape (T', E, 3)
        Returns:
            Dictionary of metrics
        """
        hit_acc = 0.0
        hit_ppv = 0.0
        hit_tpr = 0.0
        hit_f1 = 0.0
        velocity_mse = 0.0
        offset_mse = 0.0

        for reference, hypothesis in zip(references, hypotheses):
            hit_pred_int = hypothesis[:, :, 0].int()
            h_true_int   = reference[:, :, 0].int()

            hit_tp = ((hit_pred_int == 1) & (h_true_int == 1)).sum().item()
            hit_fp = ((hit_pred_int == 1) & (h_true_int == 0)).sum().item()
            hit_fn = ((hit_pred_int == 0) & (h_true_int == 1)).sum().item()
            hit_tn = ((hit_pred_int == 0) & (h_true_int == 0)).sum().item()

            hit_acc += (hit_tp + hit_tn) / (hit_tp + hit_fp + hit_fn + hit_tn) if (hit_tp + hit_fp + hit_fn + hit_tn) > 0 else 0.0
            hit_ppv += hit_tp / (hit_tp + hit_fp) if (hit_tp + hit_fp) > 0 else 0.0
            hit_tpr += hit_tp / (hit_tp + hit_fn) if (hit_tp + hit_fn) > 0 else 0.0
            hit_f1  += (2 * hit_tp) / (2 * hit_tp + hit_fp + hit_fn) if (2 * hit_tp + hit_fp + hit_fn) > 0 else 0.0
            
            velocity_mse += F.mse_loss(reference[:, :, 1], hypothesis[:, :, 1]).item()
            offset_mse += F.mse_loss(reference[:, :, 2], hypothesis[:, :, 2]).item()

        hit_acc /= len(references)
        hit_ppv /= len(references)
        hit_tpr /= len(references)
        hit_f1  /= len(references)
        velocity_mse /= len(references)
        offset_mse /= len(references)

        return {
            'hit_acc': hit_acc,
            'hit_ppv': hit_ppv,
            'hit_tpr': hit_tpr,
            'hit_f1': hit_f1,
            'velocity_mse': velocity_mse,
            'offset_mse': offset_mse
        }
    
    def _save_plots(self, train_plots, val_results, epoch, num_samples: int = 3):
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

        # TODO: Save generated midi
        for result in val_results:
            result['generated_sample'].dump(midi_dir / f"{result['generated_sample'].id}.mid")
        pass
