from .base_trainer import BaseTrainer
from data import DrumMIDIFeature
from typing import Dict, Any, Optional, List
import torch
import torch.nn as nn
from tqdm import tqdm
import torch.nn.functional as F
from models import GrooveIQ
import matplotlib.pyplot as plt
import pickle
from utils import DrumMetrics, RunningAverageMeter

class GrooveIQ_Trainer(BaseTrainer):
    """
    GrooveIQ Trainer class that handles training, validation, and recognition loops.

    This trainer implements:
    1. Training loop with gradient accumulation, mixed precision training
    2. Validation loop for model evaluation
    3. Recognition capabilities with different decoding strategies (greedy, beam search)
    """
    def __init__(self, model : GrooveIQ, config : Dict[str, Any], run_name : str, config_file : str, device : str = None):
        super().__init__(model, config, run_name, config_file, device)
        
        #  Initialize Hit loss
        self.pos_weight = config['loss'].get('pos_weight', 17.0)
        self.hit_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=torch.tensor(self.pos_weight))

        #  Initialize Velocity loss
        self.velocity_loss = nn.MSELoss(reduction='none')

        #  Initialize Offset loss
        self.offset_loss = nn.MSELoss(reduction='none')

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

        running_metrics = {
            "hit_bce": 0.0,
            "velocity_mse": 0.0,
            "velocity_mae": 0.0,
            "velocity_corr": 0.0,
            "velocity_range_diff": 0.0,
            "offset_mse": 0.0,
            "offset_mae": 0.0,
            "offset_tightness": 0.0,
            "offset_ahead": 0.0,
            "offset_behind": 0.0,
            "velocity_penalty": 0.0,
            "offset_penalty": 0.0,
            "joint_loss": 0.0,
            "hit_acc": 0.0,
            "hit_ppv": 0.0,
            "hit_tpr": 0.0,
            "hit_f1": 0.0,
            "hit_perplexity": 0.0,
        }
        running_sample_count = 0

        self.optimizer.zero_grad()

        for i, batch in enumerate(dataloader):
            grids, samples = batch['grid'].to(self.device), batch['samples']
            h_true, v_true, o_true = grids[:, :, :, 0], grids[:, :, :, 1], grids[:, :, :, 2]

            with torch.autocast(device_type=self.device, dtype=torch.float16):
                h_logits, v_pred, o_pred, latent, button_hvo, velocity_penalty, offset_penalty = self.model(grids)

                hit_penalty = torch.where(h_true == 1, self.hit_penalty, 0.0)

                hit_bce = self.hit_loss(h_logits, h_true).mean()
                velocity_mse = (self.velocity_loss(v_pred, v_true) * hit_penalty).mean()
                offset_mse = (self.offset_loss(o_pred, o_true) * hit_penalty).mean()
                
                joint_loss = hit_bce + velocity_mse + offset_mse + velocity_penalty + offset_penalty

            # Metrics
            batch_size = grids.size(0)
            metrics = DrumMetrics.hit_metrics((torch.sigmoid(h_logits) > 0.5).int(), h_true.int())
            hit_perplexity = DrumMetrics.perplexity(hit_bce)
            velocity_mae = DrumMetrics.mae_metrics(v_pred, v_true)
            velocity_corr = DrumMetrics.pearson_corr(v_pred, v_true)
            velocity_range_diff = DrumMetrics.range_diff(v_pred, v_true)
            offset_mae = DrumMetrics.mae_metrics(o_pred, o_true)
            offset_tightness = DrumMetrics.percent_within_tolerance(o_pred, o_true, tolerance=0.02)
            offset_push_lag = DrumMetrics.ahead_behind_ratio(o_pred, o_true)

            # Accumulate metrics
            running_sample_count += batch_size
            running_metrics["hit_bce"] += hit_bce.item() * batch_size
            running_metrics["velocity_mse"] += velocity_mse.item() * batch_size
            running_metrics["velocity_mae"] += velocity_mae * batch_size
            running_metrics["velocity_corr"] += velocity_corr * batch_size
            running_metrics["velocity_range_diff"] += velocity_range_diff * batch_size
            running_metrics["offset_mse"] += offset_mse.item() * batch_size
            running_metrics["offset_mae"] += offset_mae * batch_size
            running_metrics["offset_tightness"] += offset_tightness * batch_size
            running_metrics["offset_ahead"] += offset_push_lag["ahead"] * batch_size
            running_metrics["offset_behind"] += offset_push_lag["behind"] * batch_size
            running_metrics["velocity_penalty"] += velocity_penalty.item() * batch_size
            running_metrics["offset_penalty"] += offset_penalty.item() * batch_size
            running_metrics["joint_loss"] += joint_loss.item() * batch_size
            running_metrics["hit_acc"] += metrics["acc"] * batch_size
            running_metrics["hit_ppv"] += metrics["ppv"] * batch_size
            running_metrics["hit_tpr"] += metrics["tpr"] * batch_size
            running_metrics["hit_f1"] += metrics["f1"] * batch_size
            running_metrics["hit_perplexity"] += hit_perplexity * batch_size

            # Gradient accumulation
            scaled_loss = joint_loss / self.config['training']['gradient_accumulation_steps']
            self.scaler.scale(scaled_loss).backward()

            if (i + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                self.scaler.step(self.optimizer)
                if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step()
                self.scaler.update()
                self.optimizer.zero_grad()

            # Progress
            avg_joint_loss = running_metrics["joint_loss"] / running_sample_count
            avg_hit_acc = running_metrics["hit_acc"] / running_sample_count
            avg_hit_f1 = running_metrics["hit_f1"] / running_sample_count

            batch_bar.set_postfix(
                joint=f"{avg_joint_loss:.4f}",
                h_acc=f"{avg_hit_acc:.4f}",
                h_f1=f"{avg_hit_f1:.4f}",
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

        # Final metrics
        avg_metrics = {k: v / running_sample_count for k, v in running_metrics.items()}
        batch_bar.close()
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
        # Generate
        results = self.generate(dataloader, num_batches=num_batches)
        
        # Extract references and hypotheses from results
        references = [result['target_grid'] for result in results]
        hypotheses = [result['generated_grid'] for result in results]
        
        # Calculate metrics on full batch
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
            val_metrics, val_results = self._validate_epoch(val_dataloader, num_batches=5)

            # Step ReduceLROnPlateau scheduler with validation loss
            if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_metrics['hit_acc'])
            
            # Log metrics
            metrics = {
                'train': train_metrics,
                'val': val_metrics
            }
            self._log_metrics(metrics, epoch)

            # Generate and save plots
            self._save_plots(train_plots, val_results, epoch)
            
            # Save generated midi
            self._save_midi(val_results, epoch)

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
                latent = self.model.encode(grids)

                # Get button HVO
                button_hvo = self.model.make_button_hvo(latent)

                # Generate
                generated_grids = self.model.generate(button_hvo, max_steps=max_length)

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
                        "button_hvo": button_hvo[b, :, :, :].cpu().detach(),
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
        hit_acc = hit_ppv = hit_tpr = hit_f1 = 0.0
        velocity_mse = velocity_mae = velocity_corr = velocity_range_diff = 0.0
        offset_mse = offset_mae = offset_tightness = offset_ahead = offset_behind = 0.0

        for ref, hyp in zip(references, hypotheses):
            metrics = DrumMetrics.hit_metrics(hyp[:, :, 0].int(), ref[:, :, 0].int())
            hit_acc += metrics["acc"]
            hit_ppv += metrics["ppv"]
            hit_tpr += metrics["tpr"]
            hit_f1  += metrics["f1"]

            v_ref, v_hyp = ref[:, :, 1], hyp[:, :, 1]
            velocity_mse += DrumMetrics.mse_metrics(v_hyp, v_ref)
            velocity_mae += DrumMetrics.mae_metrics(v_hyp, v_ref)
            velocity_corr += DrumMetrics.pearson_corr(v_hyp, v_ref)
            velocity_range_diff += DrumMetrics.range_diff(v_hyp, v_ref)

            o_ref, o_hyp = ref[:, :, 2], hyp[:, :, 2]
            offset_mse += DrumMetrics.mse_metrics(o_hyp, o_ref)
            offset_mae += DrumMetrics.mae_metrics(o_hyp, o_ref)
            offset_tightness += DrumMetrics.percent_within_tolerance(o_hyp, o_ref)
            offset_push_lag = DrumMetrics.ahead_behind_ratio(o_hyp, o_ref)
            offset_ahead += offset_push_lag["ahead"]
            offset_behind += offset_push_lag["behind"]

        N = len(references)
        metrics_dict = {
            'hit_acc': hit_acc / N,
            'hit_ppv': hit_ppv / N,
            'hit_tpr': hit_tpr / N,
            'hit_f1': hit_f1 / N,
            'velocity_mse': velocity_mse / N,
            'velocity_mae': velocity_mae / N,
            'velocity_corr': velocity_corr / N,
            'velocity_range_diff': velocity_range_diff / N,
            'offset_mse': offset_mse / N,
            'offset_mae': offset_mae / N,
            'offset_tightness': offset_tightness / N,
            'offset_ahead': offset_ahead / N,
            'offset_behind': offset_behind / N,
        }
        return metrics_dict
    
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

        # Plot Training Grid Plot + Button HVO
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
                'button_hvo': result['button_hvo'],
            })
        
        with open(midi_dir / f"val_results_{epoch}.pkl", "wb") as f:
            pickle.dump(results, f)
