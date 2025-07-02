import pickle
import matplotlib.pyplot as plt
import numpy as np
import random
from typing import Optional, List, Union, Tuple, Dict
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from scipy.stats import gaussian_kde
import itertools
from .feature import DrumMIDIFeature
from .utils import get_num_bars
from .drum_maps import CANONICAL_DRUM_MAP
import torch
from tqdm import tqdm

@dataclass
class SampleData:
    """Container for a single drum MIDI sample and associated metadata."""
    id: str
    map: str
    style: str
    time_signature: str
    type: str
    metadata: list[str]
    midi_bytes: bytes
    num_bars: int
    feature: DrumMIDIFeature

    def __str__(self) -> str:
        return f"SampleData(id={self.id}, map={self.map}, style={self.style}, time_signature={self.time_signature}, type={self.type}, metadata={self.metadata}, num_bars={self.num_bars})"
    
    def __repr__(self) -> str:
        return f"SampleData(id={self.id}, map={self.map}, style={self.style}, time_signature={self.time_signature}, type={self.type}, metadata={self.metadata}, num_bars={self.num_bars})"
    
    def from_fixed_grid(self, grid: torch.Tensor, steps_per_quarter: int) -> "SampleData":
        """Create a sample from a fixed grid."""
        feature = self.feature.from_fixed_grid(grid, steps_per_quarter)
        numerator, denominator = map(int, self.time_signature.split('/'))
        num_bars = get_num_bars(duration=feature.end, time_signature=(numerator, denominator), tpq=self.feature.score.tpq)
        return SampleData(
            id=self.id,
            map=self.map,
            style=self.style,
            time_signature=self.time_signature,
            type=self.type,
            metadata=self.metadata,
            midi_bytes=feature.score.dumps_midi(),
            num_bars=num_bars,
            feature=feature,
        )
    
    def split_segments(self, num_bars: int) -> Tuple[List["SampleData"], int]:
        """Split the sample into segments."""
        numerator, denominator = map(int, self.time_signature.split('/'))
        time_signature = (numerator, denominator)
        segments, num_errors = self.feature.split_segments(time_signature, num_bars)
        return [
            SampleData(
                id=self.id,
                map=self.map,
                style=self.style,
                time_signature=self.time_signature,
                type=self.type,
                metadata=self.metadata,
                midi_bytes=segment.score.dumps_midi(),
                num_bars=get_num_bars(duration=segment.end, time_signature=time_signature, tpq=segment.score.tpq),
                feature=segment,
            ) for segment in segments
        ], num_errors

    def get_random_segment(self, num_bars: int) -> "SampleData":
        """Get a random segment of the sample."""
        time_sig_tuple = tuple(map(int, self.time_signature.split('/')))
        feature_segment = self.feature.get_random_segment(time_sig_tuple, num_bars)
        return SampleData(
            id=self.id,
            map=self.map,
            style=self.style,
            time_signature=self.time_signature,
            type=self.type,
            metadata=self.metadata,
            midi_bytes=feature_segment.score.dumps_midi(),
            num_bars=num_bars,
            feature=feature_segment,
        )
    
    def to_dict(self) -> dict:
        """Convert the sample to a dictionary."""
        return {
            "id": self.id,
            "map": self.map,
            "style": self.style,
            "time_signature": self.time_signature,
            "type": self.type,
            "metadata": self.metadata,
            "num_bars": self.num_bars,
            "midi_bytes": self.midi_bytes,
        }
    
    @staticmethod
    def from_dict(data: dict, drum_map: dict = None) -> "SampleData":
        """Create a sample from a dictionary."""
        return SampleData(
            id=data['id'],
            map=data['map'],
            style=data['style'],
            time_signature=data['time_signature'],
            type=data['type'],
            metadata=data['metadata'],
            midi_bytes=data['midi_bytes'],
            num_bars=data['num_bars'],
            feature=DrumMIDIFeature(data['midi_bytes']),
        )
    
    def dump(self, path: str) -> None:
        """Dump the sample to a MIDI file."""
        with open(path, "w") as f:
            pickle.dump({
                "id": self.id,
                "map": self.map,
                "style": self.style,
                "time_signature": self.time_signature,
                "type": self.type,
                "metadata": self.metadata,
                "num_bars": self.num_bars,
                "midi_bytes": self.midi_bytes,
            }, f)
    

@dataclass
class DataStats:
    """Tracks statistics for a dataset grouped by musical style."""
    name: str = "[NOT-SET]"
    num_samples: int = 0
    total_bars: int = 0
    all_samples: list[SampleData] = field(default_factory=list)
    drum_classes: Dict[int, str] = field(default_factory=lambda: {k: v for k, v in CANONICAL_DRUM_MAP.items()})
    max_hits_per_class: Dict[int, int] = field(default_factory=lambda: defaultdict(int))
    style_map: Dict[str, Dict[str, Union[int, Dict[str, int]]]] = field(default_factory=dict)
    num_neg_fixed = 0
    num_pos_fixed = 0
    num_neg_flexible = 0
    num_pos_flexible = 0

    def set_name(self, name: str) -> None:
        """Set the name of the dataset."""
        self.name = name

    def __str__(self) -> str:
        """String representation of the dataset."""
        return f"DataStats('{self.name}': {self.num_samples} samples, {self.total_bars} bars)"

    def __repr__(self) -> str:
        """Representation of the dataset."""
        return f"DataStats(name='{self.name}', num_samples={self.num_samples}, total_bars={self.total_bars})"
    
    def __len__(self) -> int:
        """Get the number of samples in the dataset."""
        return self.num_samples
    
    def __getitem__(self, ind: int) -> SampleData:
        """Get a sample from the dataset."""
        return self.all_samples[ind]

    #### ACCUMULATORS #########################################################

    def accumulate_sample(self, sample: SampleData) -> None:
        """Accumulate statistics for a single sample."""
        self.num_samples += 1
        style = sample.style
        time_sig = sample.time_signature
        sample_type = sample.type

        if style not in self.style_map:
            self.style_map[style] = {
                "sample_count": 0,
                "bar_count": 0,
                "time_signatures": defaultdict(int),
                "types": defaultdict(int),
                "bars": defaultdict(int)
            }

        style_entry = self.style_map[style]
        style_entry["sample_count"] += 1
        style_entry["time_signatures"][time_sig] += 1
        style_entry["types"][sample_type] += 1
        style_entry["bar_count"] += sample.num_bars
        style_entry["bars"][sample.num_bars] += 1
        self.total_bars += sample.num_bars
        self._compute_max_hits_per_class(sample.feature)
        self.all_samples.append(sample)

    def accumulate_dict(self, data: dict) -> None:
        """
        Accumulate statistics for a single sample from a dictionary.
        Args:
            data: Dictionary containing sample metadata and MIDI bytes.
            Format:
                    {
                        'id': int,
                        'map': str,
                        'style': str,
                        'time_signature': str,
                        'type': str,
                        'metadata': List[str],
                        'midi_bytes': bytes
                    }
        """
        self.num_samples += 1
        style = data['style']
        time_sig = data['time_signature']
        sample_type = data['type']

        if style not in self.style_map:
            self.style_map[style] = {
                "sample_count": 0,
                "bar_count": 0,
                "time_signatures": defaultdict(int),
                "types": defaultdict(int),
                "bars": defaultdict(int)
            }

        style_entry = self.style_map[style]
        style_entry["sample_count"] += 1
        style_entry["time_signatures"][time_sig] += 1
        style_entry["types"][sample_type] += 1

        # Compute number of bars
        feature = DrumMIDIFeature(data['midi_bytes'])
        self._compute_max_hits_per_class(feature)
        numerator, denominator = map(int, time_sig.split('/'))
        num_bars = get_num_bars(duration=feature.end, time_signature=(numerator, denominator), tpq=feature.score.tpq)

        style_entry["bar_count"] += num_bars
        style_entry["bars"][num_bars] += 1
        self.total_bars += num_bars

        self.all_samples.append(SampleData(
            id=data['id'],
            map=data['map'],
            style=style,
            time_signature=time_sig,
            type=sample_type,
            metadata=data['metadata'],
            midi_bytes=data['midi_bytes'],
            num_bars=num_bars,
            feature=feature,
        ))

    def get_pos_neg_counts(self, steps_per_quarter: int) -> Tuple[int, int]:
        """Get the number of positive and negative samples in the dataset."""
        for sample in self.all_samples:
            fixed_grid, _ = sample.feature.to_fixed_grid(steps_per_quarter=steps_per_quarter)
            flexible_grid, _ = sample.feature.to_flexible_grid(self.max_hits_per_class, steps_per_quarter=steps_per_quarter)
            self.num_neg_fixed += (fixed_grid == 0).sum()
            self.num_pos_fixed += (fixed_grid == 1).sum()
            self.num_neg_flexible += (flexible_grid == 0).sum()
            self.num_pos_flexible += (flexible_grid == 1).sum()
        return self.num_neg_fixed, self.num_pos_fixed, self.num_neg_flexible, self.num_pos_flexible


    #### QUERYING/FILTERING #########################################################

    def query(
        self,
        styles: Optional[List[str]] = None,
        time_signatures: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        num_bars: Optional[Union[List[int], Tuple[int, int]]] = None
    ) -> List[SampleData]:
        """
        Query the dataset with the given filters.
        
        Args:
            styles: List of styles to filter by.
            time_signatures: List of time signatures to filter by.
            types: List of sample types to filter by.
            num_bars: Either a list of exact bar counts or a tuple (min_bars, max_bars) for a range query.
        
        Returns:
            List of samples matching the given filters.
        """
        # === Validate styles ===
        if styles is not None:
            missing = set(styles) - set(self.style_map.keys())
            if missing:
                raise ValueError(f"Some styles not found: {missing}")
        
        # === Validate time_signatures ===
        if time_signatures is not None:
            all_sigs = set()
            for style in (styles or self.style_map.keys()):
                all_sigs.update(self.style_map[style]["time_signatures"].keys())
            missing = set(time_signatures) - all_sigs
            if missing:
                raise ValueError(f"Some time signatures not found: {missing}")

        # === Validate types ===
        if types is not None:
            all_types = set()
            for style in (styles or self.style_map.keys()):
                all_types.update(self.style_map[style]["types"].keys())
            missing = set(types) - all_types
            if missing:
                raise ValueError(f"Some types not found: {missing}")

        # === Validate num_bars ===
        if isinstance(num_bars, list):
            all_bars = set()
            for style in (styles or self.style_map.keys()):
                all_bars.update(self.style_map[style]["bars"].keys())
            missing = set(num_bars) - all_bars
            if missing:
                raise ValueError(f"Some bar counts not found: {missing}")
        elif isinstance(num_bars, tuple):
            if len(num_bars) != 2 or not all(isinstance(x, int) for x in num_bars):
                raise ValueError("num_bars tuple must be (min_bars, max_bars)")

        # === Apply Filters ===
        def sample_matches(sample):
            return (
                (styles is None or sample.style in styles) and
                (time_signatures is None or sample.time_signature in time_signatures) and
                (types is None or sample.type in types) and
                (
                    num_bars is None or (
                        isinstance(num_bars, list) and sample.num_bars in num_bars
                    ) or (
                        isinstance(num_bars, tuple) and num_bars[0] <= sample.num_bars <= num_bars[1]
                    )
                )
            )

        return [sample for sample in self.all_samples if sample_matches(sample)]
    
    def filter(
        self,
        styles: Optional[List[str]] = None,
        time_signatures: Optional[List[str]] = None,
        types: Optional[List[str]] = None,
        num_bars: Optional[Union[List[int], Tuple[int, int]]] = None
    ) -> 'DataStats':
        """
        Create a new DataStats object containing only samples that match the given filters.
        
        Args:
            styles: List of styles to filter by.
            time_signatures: List of time signatures to filter by.
            types: List of sample types to filter by.
            num_bars: Either a list of exact bar counts or a tuple (min_bars, max_bars) for a range query.
        
        Returns:
            A new DataStats object containing only the filtered samples.
        """
        # Get filtered samples using existing query method
        filtered_samples = self.query(styles, time_signatures, types, num_bars)
        
        # Create new DataStats object
        filtered_stats = DataStats()
        suffix = self._make_filter_suffix(styles, time_signatures, types, num_bars)
        filtered_stats.name = f"{self.name}_{suffix}"
        for sample in filtered_samples:
            filtered_stats.accumulate_sample(sample)
        return filtered_stats
    
    def stratified_split(self, train_size: float, val_size: float, test_size: float) -> Tuple['DataStats', 'DataStats', 'DataStats']:
        """
        Split the dataset into train, validation, and test sets.
        Preserves the percentage of number of bars per style in the split datasets.

        Args:
            train_size: Fraction of data for training (e.g., 0.7)
            val_size: Fraction of data for validation (e.g., 0.15)
            test_size: Fraction of data for testing (e.g., 0.15)
            
        Returns:
            Tuple of (train_stats, val_stats, test_stats) DataStats objects
            
        Raises:
            ValueError: If sizes don't sum to 1.0 or if any size is negative
        """
        # Validate input parameters
        total_size = train_size + val_size + test_size
        if abs(total_size - 1.0) > 1e-6:
            raise ValueError(f"Split sizes must sum to 1.0, got {total_size}")
        if any(size < 0 for size in [train_size, val_size, test_size]):
            raise ValueError("All split sizes must be non-negative")
        
        # Get all styles and their bar counts
        styles = list(self.style_map.keys())
        style_bar_counts = {style: self.style_map[style]['bar_count'] for style in styles}
        
        # Calculate per split per style bar counters
        train_bar_counts = {style: {'curr_bars': 0, 'target_bars': int(style_bar_counts[style] * train_size)} for style in styles}
        val_bar_counts   = {style: {'curr_bars': 0, 'target_bars': int(style_bar_counts[style] * val_size)} for style in styles}
        test_bar_counts  = {style: {'curr_bars': 0, 'target_bars': int(style_bar_counts[style] * test_size)} for style in styles}

         # Group samples by style
        style_samples = {style: [] for style in styles}
        for sample in self.all_samples:
            style_samples[sample.style].append(sample)

        # Shuffle samples *per style*
        for style in styles:
            random.shuffle(style_samples[style])

        # Create new DataStats objects
        split_id = f"tr{train_size:.2f}-va{val_size:.2f}-te{test_size:.2f}"
        train_stats = DataStats()
        train_stats.set_name(f"{self.name}_{split_id}_train")
        val_stats = DataStats()
        val_stats.set_name(f"{self.name}_{split_id}_val")
        test_stats = DataStats()
        test_stats.set_name(f"{self.name}_{split_id}_test")
        
        # Split samples style by style
        for style in styles:
            for sample in style_samples[style]:
                num_bars = sample.num_bars

                if train_bar_counts[style]['curr_bars'] + num_bars <= train_bar_counts[style]['target_bars']:
                    train_bar_counts[style]['curr_bars'] += num_bars
                    train_stats.accumulate_sample(sample)
                elif val_bar_counts[style]['curr_bars'] + num_bars <= val_bar_counts[style]['target_bars']:
                    val_bar_counts[style]['curr_bars'] += num_bars
                    val_stats.accumulate_sample(sample)
                else:
                    test_bar_counts[style]['curr_bars'] += num_bars
                    test_stats.accumulate_sample(sample)

        return train_stats, val_stats, test_stats
    
    from tqdm import tqdm

    def split_segments(self, num_bars: int) -> "DataStats":
        """Split the dataset into segments and show progress."""
        split_stats = DataStats()
        split_stats.set_name(f"{self.name}_{num_bars}bar")
        num_accumulated = 0
        num_errors = 0

        pbar = tqdm(self.all_samples, desc="Splitting samples", unit="sample")

        for sample in pbar:
            try:
                segments, sample_errors = sample.split_segments(num_bars=num_bars)
            except Exception as e:
                # Count as 1 error if the whole sample failed
                num_errors += 1
                pbar.set_postfix({
                    "accumulated": num_accumulated,
                    "errors": num_errors
                })
                continue

            num_errors += sample_errors

            for segment in segments:
                try:
                    split_stats.accumulate_sample(segment)
                    num_accumulated += 1
                except Exception as e:
                    num_errors += 1

            # Update progress bar postfix
            pbar.set_postfix({
                "accumulated": num_accumulated,
                "errors": num_errors
            })

        pbar.close()
        return split_stats

    
    #### I/O #########################################################
    
    def serialize(self, path: str) -> None:
        """Serialize the dataset to a pickle file."""
        serializable_samples = []
        for sample in self.all_samples:
            serializable_samples.append({
                'id': sample.id,
                'map': sample.map,
                'style': sample.style,
                'time_signature': sample.time_signature,
                'type': sample.type,
                'metadata': sample.metadata,
                'midi_bytes': sample.midi_bytes,
            })
        with open(path, "wb") as f:
            pickle.dump(serializable_samples, f)

    #### SUMMARIZATION/VISUALIZATION #########################################################
    
    def summarize(self, verbose: bool = False) -> None:
        """Prints a summary of dataset statistics with improved formatting."""
        import pprint

        print(f"\nSummary for: '{self.name}'")
        print("-" * 60)
        print(f"Total samples     : {self.num_samples}")
        print(f"Total bars        : {self.total_bars}")
        print(f"Number of styles  : {len(self.style_map)}")

        # Max hits per class string
        max_hits_str = "\n".join(
            f"    • {name:<12} [MIDI {pitch}] → {self.max_hits_per_class[pitch]}"
            for pitch, name in self.drum_classes.items()
            if self.max_hits_per_class[pitch] > 0
        )
        print(f"Max hits per class:\n{max_hits_str}")

        print("\nPer-style breakdown:")
        for i, (style, stats) in enumerate(self.style_map.items()):
            sample_count = stats["sample_count"]
            bar_count = stats["bar_count"]
            avg_bars = bar_count / sample_count if sample_count else 0

            print(f"\n  {style}")
            print(f"    • Samples       : {sample_count}")
            print(f"    • Total Bars    : {bar_count}")
            print(f"    • Avg Bars/Sample : {avg_bars:.2f}")

            if verbose:
                def format_dict(d, indent=4):
                    lines = [f"{' ' * indent}{k}: {v}" for k, v in sorted(d.items())]
                    return "\n".join(lines)
                ts_str = format_dict(dict(stats["time_signatures"]))
                type_str = format_dict(dict(stats["types"]))
                bar_len_str = format_dict(dict(sorted(stats["bars"].items())))

                print(f"    • Time Signatures : {ts_str}")
                print(f"    • Types           : {type_str}")
                print(f"    • Bar Lengths     : {bar_len_str}")

    def visualize(self) -> None:
        """
        Visualizes various statistics from the dataset using modern matplotlib styling.

        Generates a 3x2 grid of plots including:
        - Samples per style (horizontal bar)
        - Time signature distribution (vertical bar)
        - Total bars per style (vertical bar)
        - Average bars per sample (vertical bar)
        - Distribution of bar lengths (histogram + KDE)
        - Beat vs Fill type distribution (donut pie chart)

        Notes:
            - Applies a clean, readable theme using plt.rcParams.
            - Adds labels and legends for interpretability.
            - Requires `scipy` for KDE; skips if unavailable.
        """
        # --- Style Setup ---
        plt.style.use('default')
        plt.rcParams.update({
            'font.size': 11,
            'axes.titlesize': 13,
            'axes.labelsize': 11,
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'font.family': 'DejaVu Sans',
            'axes.facecolor': '#f8f9fa',
            'figure.facecolor': 'white',
            'axes.grid': True,
            'grid.alpha': 0.3,
            'grid.linestyle': '--'
        })

        styles = list(self.style_map.keys())
        modern_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#FF8A80', '#80DEEA']
        color_cycle = itertools.cycle(modern_colors)
        fig, axes = plt.subplots(3, 2, figsize=(20, 16), gridspec_kw={'height_ratios': [1.1, 1, 1]})
        fig.suptitle(f'Dataset Analysis: {self.name}', fontsize=18, fontweight='bold', y=0.95)

        # Helper styling
        label_kwargs = dict(fontweight='bold', fontsize=11)
        for row in axes:
            for ax in row:
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

        # --- Samples per Style (horizontal bar) ---
        counts = [self.style_map[style]['sample_count'] for style in styles]
        total_samples = sum(counts)
        percentages = [count / total_samples * 100 for count in counts]
        bar_colors = [next(color_cycle) for _ in styles]
        bars = axes[0, 0].barh(styles, percentages, color=bar_colors, alpha=0.8, edgecolor='white', linewidth=1)
        axes[0, 0].bar_label(bars, fmt='%.1f', padding=5, fontsize=10, fontweight='bold')
        axes[0, 0].set_title("Samples per Style", **label_kwargs)
        axes[0, 0].set_xlabel("Sample Count (%)", **label_kwargs)
        axes[0, 0].invert_yaxis()
        axes[0, 0].grid(axis='x')

        # --- Time Signature Distribution (vertical bar) ---
        time_sig_counts = sum([Counter(self.style_map[style]['time_signatures']) for style in styles], Counter())
        total_time_sigs = sum(time_sig_counts.values())
        time_sig_percentages = [count / total_time_sigs * 100 for count in time_sig_counts.values()]
        bars = axes[0, 1].bar(
            range(len(time_sig_counts)),
            time_sig_percentages,
            color=modern_colors[:len(time_sig_counts)],
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        axes[0, 1].bar_label(bars, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')
        axes[0, 1].set_title("Time Signature Distribution", **label_kwargs)
        axes[0, 1].set_xticks(range(len(time_sig_counts)))
        axes[0, 1].set_xticklabels(list(time_sig_counts.keys()), rotation=45, ha='right')
        axes[0, 1].set_ylabel("Count (%)", **label_kwargs)
        axes[0, 1].grid(axis='y')

        # --- Total Bars per Style ---
        total_bars = [self.style_map[style]['bar_count'] for style in styles]
        total_all_bars = sum(total_bars)
        bar_percentages = [bars / total_all_bars * 100 for bars in total_bars]
        bars = axes[1, 0].bar(
            styles,
            bar_percentages,
            color=modern_colors[:len(styles)],
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        axes[1, 0].bar_label(bars, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')
        axes[1, 0].set_title("Total Bars per Style", **label_kwargs)
        axes[1, 0].set_ylabel("Total Bars (%)", **label_kwargs)
        axes[1, 0].set_xticks(range(len(styles)))
        axes[1, 0].set_xticklabels(styles, rotation=45, ha='right')
        axes[1, 0].grid(axis='y')

        # --- Average Bars per Sample ---
        avg_bars = [self.style_map[style]['bar_count'] / self.style_map[style]['sample_count'] for style in styles]
        bars = axes[1, 1].bar(
            styles,
            avg_bars,
            color=modern_colors[:len(styles)],
            alpha=0.8,
            edgecolor='white',
            linewidth=1
        )
        axes[1, 1].bar_label(bars, fmt='%.1f', padding=3, fontsize=10, fontweight='bold')
        axes[1, 1].set_title("Average Bars per Sample", **label_kwargs)
        axes[1, 1].set_ylabel("Average Bars", **label_kwargs)
        axes[1, 1].set_xticks(range(len(styles)))
        axes[1, 1].set_xticklabels(styles, rotation=45, ha='right')
        axes[1, 1].grid(axis='y')

        # --- Bar Length Histogram with KDE ---
        bar_counts = sum([Counter(self.style_map[style]['bars']) for style in styles], Counter())
        bar_list = np.repeat(list(bar_counts.keys()), list(bar_counts.values()))
        n_bins = min(15, len(set(bar_list)))
        n, bins, patches = axes[2, 0].hist(
            bar_list,
            bins=n_bins,
            color='#4ECDC4',
            alpha=0.7,
            edgecolor='white',
            linewidth=1
        )
        if len(bar_list) > 1:
            kde = gaussian_kde(bar_list)
            x_range = np.linspace(min(bar_list), max(bar_list), 100)
            bin_width = bins[1] - bins[0]
            kde_curve = kde(x_range) * len(bar_list) * bin_width
            axes[2, 0].plot(x_range, kde_curve, color='#FF6B6B', linewidth=2, label='Density')
            axes[2, 0].legend()

        axes[2, 0].set_title("Distribution of Bar Lengths", **label_kwargs)
        axes[2, 0].set_xlabel("Bar Length", **label_kwargs)
        axes[2, 0].set_ylabel("Frequency", **label_kwargs)
        axes[2, 0].grid(axis='y')

        # --- Beat vs Fill Donut Chart ---
        type_counts = sum([Counter(self.style_map[style]['types']) for style in styles], Counter())
        wedges, texts, autotexts = axes[2, 1].pie(
            list(type_counts.values()),
            labels=None,
            autopct='%1.1f%%',
            startangle=90,
            colors=modern_colors[:len(type_counts)],
            wedgeprops=dict(width=0.7, edgecolor='white', linewidth=2, alpha=0.8),
            textprops=dict(fontsize=11, fontweight='bold', color='white')
        )
        axes[2, 1].set_title("Beat vs Fill Distribution", **label_kwargs)

        legend_elements = [
            plt.Rectangle((0, 0), 1, 1, facecolor=modern_colors[i], alpha=0.8)
            for i in range(len(type_counts))
        ]
        axes[2, 1].legend(
            legend_elements,
            [f"{label} ({count:,})" for label, count in type_counts.items()],
            loc='center left',
            bbox_to_anchor=(1.05, 0.5),
            borderaxespad=0.,
            title="Sample Types",
            title_fontsize=11,
            fontsize=10
        )

        # --- Layout Adjustments ---
        plt.tight_layout(rect=[0, 0, 0.95, 0.95])
        plt.show()

    #### HELPER FUNCTIONS ################################

    def _compute_max_hits_per_class(self, feature: DrumMIDIFeature, steps_per_quarter: int = 4):
        '''Utility function to compute the maximum number of hits per drum class'''
        hits_per_timestep = defaultdict(int)
        for note in feature.score.tracks[0].notes:
                if note.pitch not in self.drum_classes.keys():
                    print(f"Warning: Note {note.pitch} not in drum classes, skipping...")
                    continue
                pitch = note.pitch
                t = int(np.round((note.time / feature.score.tpq) * steps_per_quarter))
                hits_per_timestep[(t, pitch)] += 1

        for (t, pitch), count in hits_per_timestep.items():
            self.max_hits_per_class[pitch] = max(self.max_hits_per_class[pitch], count)
    
    def _make_filter_suffix(
            self,
            styles: Optional[List[str]] = None,
            time_signatures: Optional[List[str]] = None,
            types: Optional[List[str]] = None,
            num_bars: Optional[Union[List[int], Tuple[int, int]]] = None) -> str:
        '''Suffix based on query arguments'''
        parts = []
        if styles:
            parts.append(f"styles={'-'.join(styles)}")
        if time_signatures:
            ts_sanitized = [ts.replace('/', '-') for ts in time_signatures]
            parts.append(f"ts={'-'.join(ts_sanitized)}")
        if types:
            parts.append(f"types={'-'.join(types)}")
        if isinstance(num_bars, list):
            parts.append(f"bars={'-'.join(map(str, num_bars))}")
        elif isinstance(num_bars, tuple):
            parts.append(f"bars={num_bars[0]}to{num_bars[1]}")
        return "_".join(parts) if parts else "unfiltered"
