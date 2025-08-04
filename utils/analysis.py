from scipy import stats, integrate
import numpy as np
from tqdm import tqdm
import sklearn
from scipy.spatial.distance import cdist
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import yaml
from models import GrooveIQ
from data import CANONICAL_DRUM_MAP, FeatureDescriptors
import torch
import torch.nn.functional as F
from typing import Dict
from scipy.stats import pearsonr
from scipy.signal import correlate

def load_checkpoint(expt_path : str, model_name : str, device : str = "cpu"):
    """
    Loads a checkpoint from a given path
    """
    # ======= Paths =======
    chkpt_path  = os.path.join(expt_path, "checkpoints", model_name + ".pth")
    config_path = os.path.join(expt_path, "config.yaml")
    audio_save_dir = os.path.join(expt_path, "_renders")
    os.makedirs(audio_save_dir, exist_ok=True)

    # ======= Load Config =======
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # ======= Parameters =======
    # Mapping for button sequence
    fixed_grid_drum_mapping = {pitch: [i] for i, pitch in enumerate(CANONICAL_DRUM_MAP.keys())}
    MAX_LENGTH = 33
    E = 9 # Number of drum instruments
    M = 3 # Number of steps per quarter

    # ======= Model =======
    model_config = config["model"]
    model_config.update(
        T=MAX_LENGTH,
        E=E,
        M=M
    )

    model = GrooveIQ(**model_config)
    input_size = [(4, 33, 9, 3)]
    checkpoint      = torch.load(chkpt_path, map_location=device, weights_only=True)
    print(model.load_state_dict(checkpoint['model_state_dict'], strict=True))
    return model


def infer_sample(sample, model, is_heuristic=False):
    """
    Runs a sample through a model and returns the grid, generated grids, hit probabilities, and button HVO
    """
    sample, grid, button_hvo = sample['sample'], sample['grid'], sample['button_hvo']
    if grid.shape[0] < 33:
        grid = torch.cat([grid, torch.zeros((33 - grid.shape[0], 9, 3))], dim=0)
        button_hvo = torch.cat([button_hvo, torch.zeros((33 - button_hvo.shape[0], 2, 3))], dim=0)
    
    grid = grid.unsqueeze(0)
    button_hvo = button_hvo.unsqueeze(0)
    

    encoded, button_repr = model.encode(grid)
    if not is_heuristic:
        button_hits = model.make_button_hits(button_repr)
    else:
        button_hits = button_hvo[:, :, :, 0] # (B, T, num_buttons)

    button_embed = model.make_button_embed(button_hits)
    z_post, _, _ = model.make_z_post(button_embed, encoded)
    generated_grids, hit_probs = model.generate(button_embed, z_post, max_steps=33, threshold=0.5)
    generated_grids = generated_grids[:, 1:, :, :] # Drop SOS token
    generated_grids = generated_grids.squeeze(0)
    grid = grid.squeeze(0)
    button_hvo = torch.cat(
                [
                    button_hits.unsqueeze(-1), 
                    torch.ones_like(button_hits).unsqueeze(-1).repeat(1, 1, 1,1) * 0.8,  # 0.8 : velocity
                    torch.zeros_like(button_hits).unsqueeze(-1).repeat(1, 1, 1, 1),      # 0 : offset
                ], dim=-1) # (1, T, num_buttons, M)
    button_hvo = button_hvo.squeeze(0) # (T, num_buttons, M)

    return grid, generated_grids, hit_probs, button_hvo


def overlap_area(A, B):
    """Compute the overlap area between the two PDFs"""
    pdf_A = stats.gaussian_kde(A, bw_method='scott')
    pdf_B = stats.gaussian_kde(B, bw_method='scott')
    return integrate.quad(lambda x: min(pdf_A(x), pdf_B(x)), np.min((np.min(A), np.min(B))), np.max((np.max(A), np.max(B))))[0]


def kl_dist(A, B, num_sample=1000):
    """Compute the KL distance and overlap area between the two PDFs"""
    pdf_A = stats.gaussian_kde(A, bw_method='scott')
    pdf_B = stats.gaussian_kde(B, bw_method='scott')
    sample_A = np.linspace(np.min(A), np.max(A), num_sample)
    sample_B = np.linspace(np.min(B), np.max(B), num_sample)
    return stats.entropy(pdf_A(sample_A), pdf_B(sample_B))


def compute_1d_intra_distances(x, normalize=False):
    """Computes all pairwise distances within a 1D array"""
    # Calculate unique pairwise distances
    if isinstance(x, list):
        x = np.array(x)
    x = x.reshape(-1, 1)
    if normalize:
        x = sklearn.preprocessing.normalize(x, norm='l1')
    intra_distances = cdist(x, x, metric='euclidean')
    return intra_distances


def compute_1d_inter_distances(x, y, normalize=False):
    """Computes all pairwise distances between two 1D arrays"""
    if isinstance(x, list):
        x = np.array(x)
    if isinstance(y, list):
        y = np.array(y)
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    if normalize:
        x = sklearn.preprocessing.normalize(x, norm='l1')
        y = sklearn.preprocessing.normalize(y, norm='l1')
    inter_distances = cdist(x, y, metric='euclidean')
    return inter_distances


def compute_per_feature_distances(A_dict : dict, B_dict : dict, baseline_dict : dict = None):
    """Computes all pairwise distances between two sets of features"""
    per_feature_dists = {}
    for feature in tqdm(A_dict.keys(), desc="Computing per-feature distances"):
        A_feat = A_dict[feature]
        B_feat = B_dict[feature]
        per_feature_dists[feature] = {
            "A_intra": compute_1d_intra_distances(A_feat),
            "B_intra": compute_1d_intra_distances(B_feat),
            "baseline_intra": None if baseline_dict is None else compute_1d_intra_distances(baseline_dict[feature]),
            "AB_inter": compute_1d_inter_distances(A_feat, B_feat),
            "A_baseline_inter": None if baseline_dict is None else compute_1d_inter_distances(A_feat, baseline_dict[feature]),
            "B_baseline_inter": None if baseline_dict is None else compute_1d_inter_distances(B_feat, baseline_dict[feature]),
        }
    return per_feature_dists


def plot_violin_distribution(
    data_map,
    group_label="Group",
    feature="value",
    palette="Set2",
    title=None,
    title_fontsize=16,
    label_fontsize=14,
    tick_fontsize=12,
    rotate_xticks=True,
    figsize=(10, 6),
):
    """
    General-purpose violin plot for visualizing feature distributions across groups.

    Args:
        data_map (dict): Dictionary of {group_name: list of values}
        group_label (str): Label for the grouping axis (x-axis)
        feature (str): Name of the feature/metric being visualized
        palette (str or list): Seaborn color palette or list of colors
        title (str): Optional plot title. Defaults to '{feature} Distribution per {group_label}'
        title_fontsize (int): Font size for the title
        label_fontsize (int): Font size for the axis labels
        tick_fontsize (int): Font size for tick labels
        rotate_xticks (bool): Whether to rotate x-tick labels
        figsize (tuple): Figure size
    """

    # Convert to long-form DataFrame
    records = []
    for group_name, values in data_map.items():
        records.extend([{group_label: group_name, feature: v} for v in values])
    df = pd.DataFrame(records)

    # Plot
    plt.figure(figsize=figsize)
    sns.violinplot(data=df, x=group_label, y=feature, palette=palette, inner="box")

    # Labeling
    plot_title = title or f"{feature.replace('_', ' ')} Distribution per {group_label}"
    plt.title(plot_title, fontsize=title_fontsize)
    plt.xlabel(group_label, fontsize=label_fontsize)
    plt.ylabel(feature.replace("_", " "), fontsize=label_fontsize)
    if rotate_xticks:
        plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
    else:
        plt.xticks(fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)

    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


def plot_violin_distribution_wrt_gt(eval_map, palette="Set2", feature="hit_ppv",
                 title_fontsize=16, label_fontsize=14, tick_fontsize=12, legend_fontsize=12):
    """
    Plots a violin plot of evaluation metric distributions per model,
    with each model shown alongside its ground truth.

    Args:
        eval_map (dict): Dictionary with keys as model names and one key as 'Ground Truth'.
                         Each value should be a list of evaluation scores.
        palette (str or list): Seaborn color palette name or list of colors.
        feature (str): Name of the evaluation feature to plot (used for labels).
        *_fontsize (int): Font sizes for title, labels, ticks, legend.
    """
    records = []
    for model, values in eval_map.items():
        source = "Ground Truth" if model.lower() == "ground truth" else "Model"
        for val in values:
            records.append({"Model": model, "Source": source, feature: val})
    
    df = pd.DataFrame(records)

    plt.figure(figsize=(max(8, len(eval_map) * 0.8), 6))
    sns.violinplot(data=df, x="Model", y=feature, hue="Source", palette=palette, inner="box", split=True)

    # Labeling
    plt.title(f"{feature.replace('_', ' ')} Distribution per Model", fontsize=title_fontsize)
    plt.xlabel("Model", fontsize=label_fontsize)
    plt.ylabel(feature.replace("_", " "), fontsize=label_fontsize)
    plt.xticks(rotation=45, ha='right', fontsize=tick_fontsize)
    plt.yticks(fontsize=tick_fontsize)
    plt.legend(title="", fontsize=legend_fontsize)

    plt.tight_layout()
    plt.grid(axis='y')
    plt.show()


def calculate_recons_metrics(grid, generated_grids, hit_prob, threshold) -> Dict[str, float]:
    """Calculates the reconstruction metrics for a given grid, generated grids, and hit probability"""
    hit_pred_int = ((hit_prob > threshold).int()).detach().cpu()
    h_true_int   = grid[:, :, 0].int().detach().cpu()

    hit_tp = ((hit_pred_int == 1) & (h_true_int == 1)).sum().item() # True positives
    hit_fp = ((hit_pred_int == 1) & (h_true_int == 0)).sum().item()
    hit_fn = ((hit_pred_int == 0) & (h_true_int == 1)).sum().item()
    hit_tn = ((hit_pred_int == 0) & (h_true_int == 0)).sum().item()

    total    = hit_tp + hit_fp + hit_fn + hit_tn
    hit_acc  = (hit_tp + hit_tn) / total if total > 0 else 0.0
    hit_ppv  = hit_tp / (hit_tp + hit_fp) if (hit_tp + hit_fp) > 0 else 0.0
    hit_tpr  = hit_tp / (hit_tp + hit_fn) if (hit_tp + hit_fn) > 0 else 0.0
    hit_f1   = (2 * hit_tp) / (2 * hit_tp + hit_fp + hit_fn) if (2 * hit_tp + hit_fp + hit_fn) > 0 else 0.0
    
    velocity_mae = F.l1_loss(grid[:, :, 1], generated_grids[:, :, 1]).item()
    offset_mae   = F.l1_loss(grid[:, :, 2], generated_grids[:, :, 2]).item()
    return hit_acc, hit_ppv, hit_tpr, hit_f1, velocity_mae, offset_mae


def eval_recons(eval_samples, model, is_heuristic=False):
    """
    Evaluates the reconstruction metrics for a given set of samples
    """
    model.eval()
    eval_map = {
        'hit_ppv' : [],
        'hit_tpr' : [],
        'hit_f1' : [],
        'velocity_mae' : [],
        'offset_mae' : []
    }
    with torch.inference_mode():
        for sample in tqdm(eval_samples):
            grid, generated_grids, hit_probs, button_hvo = infer_sample(sample, model, is_heuristic)
            _, hit_ppv, hit_tpr, hit_f1, velocity_mae, offset_mae = calculate_recons_metrics(grid, generated_grids, hit_probs, 0.85)

            eval_map['hit_ppv'].append(hit_ppv)
            eval_map['hit_tpr'].append(hit_tpr)
            eval_map['hit_f1'].append(hit_f1)
            eval_map['velocity_mae'].append(velocity_mae)
            eval_map['offset_mae'].append(offset_mae)

    return eval_map


def eval_features(eval_samples, model, is_heuristic=False):
    model.eval()
    eval_map = {
        'total_hits' : [],
        'total_density' : [],
        'total_complexity' : [],
        'total_average_intensity' : [],
        'lowness' : [],
        'midness' : [],
        'highness' : [],
        'combined_syncopation' : [],
        'polyphonic_syncopation' : [],
        'laidbackness' : [],
        'swingness' : [],
        'timing_accuracy' : [],
    }
    with torch.inference_mode():
        for sample in tqdm(eval_samples):
            grid, generated_grids, _, _ = infer_sample(sample, model, is_heuristic)
            gt_drum_feature = sample['sample'].feature
            generated_drum_feature = gt_drum_feature.from_fixed_grid(generated_grids, steps_per_quarter=4)
            generated_descriptors = FeatureDescriptors(generated_drum_feature)
            total_hits = torch.sum(grid[:, :, 0] > 0)
            eval_map['total_hits'].append(int(total_hits.item()))
            for key in eval_map.keys():
                if key == 'total_hits':
                    continue
                eval_map[key].append(generated_descriptors.descriptors[key])
    return eval_map

def eval_alignment(eval_samples, model, is_heuristic=False):
    model.eval()
    eval_map = {
        'hit_corrs' : [],
        'peak_lags' : [],
    }
    with torch.inference_mode():
        for sample in tqdm(eval_samples):
            _, generated_grids, _, button_hvo = infer_sample(sample, model, is_heuristic)
            generated_density = generated_grids[:, :, 0].sum(dim = 1)
            button_density    = button_hvo[:, :, 0].sum(dim = 1)

            # Hit density correlation
            r, _ = pearsonr(generated_density.detach().cpu(), button_density.detach().cpu())
            eval_map['hit_corrs'].append(r)

            # Temporal cross-correlation
            generated_density_z = (generated_density - generated_density.mean()) / generated_density.std()
            button_density_z    = (button_density - button_density.mean()) / button_density.std()
            xcorr = correlate(generated_density_z, button_density_z, mode='full')
            lags = np.arange(-len(generated_density) + 1, len(generated_density))
            lag_peak = lags[np.argmax(xcorr)]
            eval_map['peak_lags'].append(lag_peak)
    return eval_map


def compute_per_feature_distances(systems: dict[str, dict], baseline_dict: dict = None):
    """
    Compute per-feature intra-system and system-vs-baseline inter distances.

    Args:
        systems (dict[str, dict]): A dictionary where each key is a system name
                                   and value is a dict mapping features to arrays.
                                   Example: {"A": A_dict, "B": B_dict}
        baseline_dict (dict, optional): Dict of baseline feature arrays. Defaults to None.

    Returns:
        dict: Nested dictionary of per-feature distances for each system.
    """
    per_feature_dists = {}
    feature_keys = next(iter(systems.values())).keys()

    for feature in tqdm(feature_keys, desc="Computing per-feature distances"):
        feature_result = {}

        for system_name, system_dict in systems.items():
            feat_data = system_dict[feature]
            feature_result[f"{system_name}_intra"] = compute_1d_intra_distances(feat_data)

            if baseline_dict is not None:
                feature_result[f"{system_name}_baseline_inter"] = compute_1d_inter_distances(
                    feat_data, baseline_dict[feature]
                )

        if baseline_dict is not None:
            feature_result["baseline_intra"] = compute_1d_intra_distances(baseline_dict[feature])

        per_feature_dists[feature] = feature_result

    return per_feature_dists


def plot_kld_oa_per_feature(
    per_feature_dists,
    system_names=None,
    feature_title_fontsize=11,
    axis_label_fontsize=12,
    tick_fontsize=10,
    legend_fontsize=10,
    suptitle_fontsize=14
):
    """
    Plots faceted KLD vs OA plots, one per feature, for multiple systems.

    Args:
        per_feature_dists (dict): Output from compute_per_feature_distances.
        system_names (list, optional): List of model names. If None, they are inferred.
        *_fontsize (int): Font sizes for various components.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns

    feature_labels = list(per_feature_dists.keys())
    num_features = len(feature_labels)

    # Dynamically infer system names
    all_keys = list(next(iter(per_feature_dists.values())).keys())
    inferred_names = sorted(set(k.split("_")[0] for k in all_keys if k.endswith("_intra") and "baseline" not in k))
    if system_names is None:
        system_names = inferred_names

    # Prepare markers and consistent colors per system
    markers = ['o', 's', '^', 'D', 'v', 'P', 'X', '*']
    marker_map = {sys: markers[i % len(markers)] for i, sys in enumerate(system_names)}
    colors = sns.color_palette("tab10", n_colors=len(system_names))
    color_map = {sys: colors[i] for i, sys in enumerate(system_names)}

    # Layout for subplots
    ncols = 3
    nrows = int(np.ceil(num_features / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*5, nrows*4), sharex=True, sharey=True)
    axes = axes.flatten()

    for i, feature in enumerate(feature_labels):
        ax = axes[i]
        data = per_feature_dists[feature]
        baseline_intra = data["baseline_intra"]
        baseline_flat = baseline_intra[np.triu_indices_from(baseline_intra, k=1)]

        for sys in system_names:
            intra = data.get(f"{sys}_intra", None)
            inter = data.get(f"{sys}_baseline_inter", None)
            if intra is None or inter is None:
                continue

            intra_flat = intra[np.triu_indices_from(intra, k=1)]

            kld = kl_dist(intra_flat, baseline_flat)
            oa = overlap_area(intra_flat, baseline_flat)

            ax.scatter(kld, oa,
                       label=sys,
                       marker=marker_map[sys],
                       color=color_map[sys],
                       s=70)
        
        ax.set_title(feature.replace("_", " ").title(), fontsize=feature_title_fontsize)
        ax.grid(True)
        ax.tick_params(axis='both', labelsize=tick_fontsize)

    # Shared axis labels
    fig.supxlabel("KL Divergence", fontsize=axis_label_fontsize)
    fig.supylabel("Overlap Area", fontsize=axis_label_fontsize)

    # Shared legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(system_names),
               frameon=False, fontsize=legend_fontsize)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    #fig.suptitle("KLD vs OA per Feature across Systems", fontsize=suptitle_fontsize)
    plt.show()

def remove_nan(model_map):
    """
    Removes nan values from a dictionary
    """
    for key, values in model_map.items():
        arr = np.array(values)
        clean_arr = arr[~np.isnan(arr)]
        model_map[key] = clean_arr
    return model_map