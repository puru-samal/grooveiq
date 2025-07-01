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
        self.temporal_loss = lambda x: ConstraintLosses().l2_temporal_diff(x)
        self.margin_loss = lambda x: ConstraintLosses().margin_loss(x)
        self.latent_loss = lambda x: ConstraintLosses().l1_sparsity(x)

        # Hit penalty
        self.hit_penalty = config['loss'].get('hit_penalty', 1.0)

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
        #running_temporal_loss = 0.0
        #running_button_penalty = 0.0
        #running_latent_penalty = 0.0
        running_velocity_penalty = 0.0
        running_offset_penalty = 0.0
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
                h_logits, v_pred, o_pred, latent, button_hvo, velocity_penalty, offset_penalty = self.model(grids)

                hit_penalty = torch.where(h_true == 1, 10.0, 0.0)

                # Hit loss
                hit_bce = self.hit_loss(h_logits, h_true).mean()

                # Velocity loss
                raw_velocity_mse = self.velocity_loss(v_pred, v_true)
                velocity_mse = (raw_velocity_mse * hit_penalty).mean()

                # Offset loss
                raw_offset_mse = self.offset_loss(o_pred, o_true)
                offset_mse = (raw_offset_mse * hit_penalty).mean()

                # Constraint losses
                # temporal_loss = self.temporal_loss(latent)
                # margin_loss = self.margin_loss(latent)

                # Button activation penalty
                #button_penalty = button_penalty.mean()
                #latent_penalty = self.latent_loss(latent)

                # Joint loss
                joint_loss = hit_bce + velocity_mse + offset_mse + velocity_penalty + offset_penalty

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
            #running_temporal_loss += temporal_loss.item() * batch_size
            #running_button_penalty += button_penalty.item() * batch_size
            running_velocity_penalty += velocity_penalty.item() * batch_size
            running_offset_penalty += offset_penalty.item() * batch_size
            #running_latent_penalty += latent_penalty.item() * batch_size
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
            #avg_temporal_loss = running_temporal_loss / running_sample_count
            #avg_button_penalty = running_button_penalty / running_sample_count
            #avg_velocity_penalty = running_velocity_penalty / running_sample_count
            #avg_offset_penalty = running_offset_penalty / running_sample_count
            #avg_latent_penalty = running_latent_penalty / running_sample_count
            avg_velocity_penalty = running_velocity_penalty / running_sample_count
            avg_offset_penalty = running_offset_penalty / running_sample_count
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
                #temporal=f"{avg_temporal_loss:.4f}",
                #button_penalty=f"{avg_button_penalty:.4f}",
                velocity_penalty=f"{avg_velocity_penalty:.4f}",
                offset_penalty=f"{avg_offset_penalty:.4f}",
                #latent_penalty=f"{avg_latent_penalty:.4f}",
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
            #'temporal_loss': running_temporal_loss / running_sample_count,
            #'button_penalty': running_button_penalty / running_sample_count,
            #'latent_penalty': running_latent_penalty / running_sample_count,
            'velocity_penalty': running_velocity_penalty / running_sample_count,
            'offset_penalty': running_offset_penalty / running_sample_count,
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
                h_true, v_true, o_true = grids[:, :, :, 0], grids[:, :, :, 1], grids[:, :, :, 2]
                
                # Get latent
                latent = self.model.encode(grids)

                # Get button HVO
                button_hvo = self.model.make_button_hvo(latent)

                # Generate
                generated_grids = self.model.sos_token.repeat(grids.size(0), 1, 1, 1) # (B, 1, E, M)  # Learned SOS token
                for t in range(max_length):
                    b = button_hvo[:, :t+1, :, :]
                    hvo_pred = self.model.generate(generated_grids, b) # (B, 1, E, 3)
                    generated_grids = torch.cat([generated_grids, hvo_pred], dim=1) # (B, T'+1, E, 3)

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

''' 
# INTERNAL USE ONLY
class ProgressiveTrainer(ASRTrainer):
    """
    Progressive Trainer class that implements curriculum learning for ASR training.

    This trainer extends ASRTrainer to implement:
    1. Stage-based training with increasing model complexity
    2. Gradual unfreezing of model layers
    3. Dynamic data subsetting
    4. Smooth transition to full model training

    Implementation Tasks:
    - TODO: Store original model layers in __init__
    - TODO: Configure model for each stage in configure_stage
    - TODO: Implement progressive training loop in progressive_train
    - TODO: Handle transition to full training in transition_to_full_training
    - TODO: Create data subsets in get_subset_dataloader

    Implementation Notes:
    1. For __init__:
        - Store original encoder and decoder layers
        - Initialize stage counter
        
    2. For configure_stage:
        - Update dropout and label smoothing
        - Activate specified encoder and decoder layers
        - Handle layer freezing based on configuration
        - Print detailed configuration information
        
    3. For progressive_train:
        - Configure model for each stage
        - Create appropriate data subset
        - Train using parent class methods
        
    4. For transition_to_full_training:
        - Restore all model layers
        - Reset loss function parameters
        - Unfreeze all parameters
        - Reset best metrics
        
    5. For get_subset_dataloader:
        - Create subset while preserving dataset attributes
        - Maintain collate function and other dataloader settings
    """
    def __init__(self, model, tokenizer, config, run_name, config_file, device=None):
        super().__init__(model, tokenizer, config, run_name, config_file, device)
        self.current_stage = 0
        # Store original layer states
        self.all_encoder_layers = list(self.model.enc_layers)
        self.all_decoder_layers = list(self.model.dec_layers)


    def configure_stage(self, stage_config):
        """Configure model for current training stage"""
        # Create a pretty header
        print("\n" + "="*80)
        print(f"Starting Stage: {stage_config['name']}".center(80))
        print("="*80)
        
        # Print key configuration details
        print(f"\nConfiguration Details:")
        print(f"├── Data Subset: {stage_config['data_subset']*100:.1f}% of training data")
        print(f"├── Training Epochs: {stage_config['epochs']}")
        print(f"├── Dropout: {stage_config['dropout']}")
        print(f"├── Label Smoothing: {stage_config['label_smoothing']}")
        
        # Update dropout and label smoothing
        self.model.dropout.p = stage_config['dropout']
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=stage_config['label_smoothing']
        )
        
        # Get freeze configurations
        encoder_freeze = stage_config.get('encoder_freeze', [])
        decoder_freeze = stage_config.get('decoder_freeze', [])
        
        # Activate and configure encoder layers
        encoder_active_layers = stage_config['encoder_active_layers']
        if encoder_freeze and len(encoder_freeze) != len(encoder_active_layers):
            raise ValueError(f"Encoder freeze list length ({len(encoder_freeze)}) must match number of active encoder layers ({len(encoder_active_layers)})")
        
        # Set the active encoder layers of the model
        self.model.enc_layers = nn.ModuleList([
            self.all_encoder_layers[i] for i in encoder_active_layers
        ])
        self.model.num_encoder_layers = len(encoder_active_layers)
        
        # Activate and configure decoder layers
        decoder_active_layers = stage_config['decoder_active_layers']
        if decoder_freeze and len(decoder_freeze) != len(decoder_active_layers):
            raise ValueError(f"Decoder freeze list length ({len(decoder_freeze)}) must match number of active decoder layers ({len(decoder_active_layers)})")
        
        # Set the active decoder layers of the model
        self.model.dec_layers = nn.ModuleList([
            self.all_decoder_layers[i] for i in decoder_active_layers
        ])
        self.model.num_decoder_layers = len(decoder_active_layers)

        # Handle layer freezing
        frozen_count = 0
        trainable_count = 0
        
        # Configure encoder layers freezing
        print("├── Encoder Layers:")
        for idx, layer in enumerate(self.model.enc_layers):
            should_freeze = encoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"│   ├── Layer {encoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")
        
        # Configure decoder layers
        print("├── Decoder Layers:")
        for idx, layer in enumerate(self.model.dec_layers):
            should_freeze = decoder_freeze[idx]
            for param in layer.parameters():
                param.requires_grad = not should_freeze
                if should_freeze:
                    frozen_count += param.numel()
                else:
                    trainable_count += param.numel()
            print(f"│   ├── Layer {decoder_active_layers[idx]}: {'Frozen' if should_freeze else 'Trainable'}")
        
        print(f"├── Frozen Parameters: {frozen_count:,}")
        print(f"└── Trainable Parameters: {trainable_count:,}")
    

    def progressive_train(self, train_dataloader, val_dataloader, stages: List[Dict[str, Any]]):
        """Progressive training through stages"""
        # Train through stages
        for stage_idx, stage_config in enumerate(stages):
            self.current_stage = stage_idx
            self.configure_stage(stage_config)
            # Get subset of train_dataloader
            subset_train_dataloader = self.get_subset_dataloader(train_dataloader, stage_config['data_subset'])
            super().train(subset_train_dataloader, val_dataloader, epochs=stage_config['epochs'])

    def transition_to_full_training(self):
        """Transition from progressive training to full training"""
        print("\n=== Transitioning to Full Training ===")
        
        # Restore all layers
        self.model.enc_layers = nn.ModuleList(self.all_encoder_layers)
        self.model.dec_layers = nn.ModuleList(self.all_decoder_layers)
        self.model.num_encoder_layers = len(self.all_encoder_layers)
        self.model.num_decoder_layers = len(self.all_decoder_layers)

        # Restore CrossEntropyLoss
        self.ce_criterion = nn.CrossEntropyLoss(
            ignore_index=self.tokenizer.pad_id,
            label_smoothing=self.config['loss']['label_smoothing']
        )
        
        # Unfreeze all parameters
        unfrozen_count = 0
        for param in self.model.parameters():
            param.requires_grad = True
            unfrozen_count += param.numel()
        print(f"├── Total Unfrozen Parameters: {unfrozen_count:,}")
        
        # Reset best metrics for new training phase
        self.best_metric = float('inf')

    
    def train(self, train_dataloader, val_dataloader, epochs):
        """
        Run full training phase.
        It is recommended to set the optimizer and scheduler explicitly before calling this function.
        like this:
        cls.optimizer = create_optimizer(self.model, self.config['optimizer'])
        cls.scheduler = create_scheduler(cls.optimizer, cls.config['scheduler'], train_dataloader)
        cls.progressive_train(train_dataloader, val_dataloader, stages)
        """
        self.transition_to_full_training()
        super().train(train_dataloader, val_dataloader, epochs=epochs)


    def get_subset_dataloader(self, dataloader, subset_fraction):
        """
        Creates a new DataLoader with a subset of the original data while preserving dataset attributes.
        
        Args:
            dataloader: Original DataLoader
            subset_fraction: Float between 0 and 1 indicating what fraction of data to keep
        
        Returns:
            New DataLoader containing only the subset of data
        """
        # Calculate how many samples we want to keep
        dataset = dataloader.dataset
        total_samples = len(dataset)
        subset_size = int(total_samples * subset_fraction)
        
        # Create random indices for the subset
        indices = torch.randperm(total_samples)[:subset_size]
        
        # Create a Subset dataset
        subset_dataset = Subset(dataset, indices)
        
        # Add necessary attributes from original dataset to subset
        subset_dataset.text_max_len = dataset.text_max_len
        subset_dataset.feat_max_len = dataset.feat_max_len
        subset_dataset.get_avg_chars_per_token = dataset.get_avg_chars_per_token
        
        # Create new DataLoader with same configuration as original
        subset_loader = torch.utils.data.DataLoader(
            subset_dataset,
            batch_size=self.config['data']['batch_size'],
            shuffle=True,
            num_workers=self.config['data']['NUM_WORKERS'],
            collate_fn=dataset.collate_fn,
            pin_memory=True
        )
        
        return subset_loader
        
'''      
        