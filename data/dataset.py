import torch
from torch.utils.data import Dataset
from .data_stats import DataStats, SampleData
import pickle
from typing import List, Dict, Literal, Tuple
from tqdm import tqdm
import random

class DrumMIDIDataset(Dataset):
    """
    A PyTorch Dataset for loading and processing DrumMIDI data.
    """
    def __init__(self, 
                 path: str, 
                 num_bars: int = 2, 
                 feature_type: Literal["fixed", "flexible"] = "fixed", 
                 steps_per_quarter: int = 4,
                 subset: float = 1.0,
                 verbose: bool = False,
        ):
        """
        Initialize the dataset.
        
        Args:
            path: Path to the pickle file containing the dataset
            num_bars: Number of bars to sample from each sample
            feature_type: Type of feature to use for the dataset [fixed, flexible]
            steps_per_quarter: Number of steps per quarter note for the fixed grid representation
            subset: Fraction of the dataset to load
            verbose: Whether to print verbose output
        """
        self.data_path = path
        self.data_stats = DataStats()
        self.data_stats.set_name(name=self.data_path.split('/')[-1].split('.')[0])
        self.num_bars = num_bars
        self.feature_type = feature_type
        self.steps_per_quarter = steps_per_quarter
        assert self.feature_type in ["fixed", "flexible"], "Invalid feature type"

        print(f"Loading dataset from: {self.data_path}...")
        with open(self.data_path, "rb") as f:
            pickle_data = pickle.load(f)
        
        if subset < 1.0:
            pickle_data = pickle_data[:int(len(pickle_data) * subset)]

        print(f"Processing {len(pickle_data)} samples...")
        num_errors = 0
        pbar = tqdm(pickle_data, desc="Accumulating:", unit="sample")
        for sample in pbar:
            try:
                self.data_stats.accumulate_dict(sample)
            except Exception as e:
                print(f"Error processing sample: {e}")
                print(f"Sample: {sample}")
                num_errors += 1
                pbar.set_postfix({"Errors": num_errors})
                continue

        print(f"Skipped {num_errors} samples due to errors.")
        print(f"Loaded and processed {len(pickle_data)} samples.\n")

        if verbose: 
            self.data_stats.summarize(verbose=False)

        self.length = len(pickle_data)
    
    
    def __len__(self) -> int:
        """
        Get the length of the dataset.
        """
        return self.length
    
    def get_max_length(self) -> int:
        """
        Get the maximum length of the dataset.
        """
        max_length = 0
        for item in self:
            max_length = max(max_length, item[1].shape[0])
        return max_length


    def __getitem__(self, ind: int) -> Tuple[SampleData, torch.Tensor, Dict]:
        """
        Get a sample from the dataset.
        Returns:
            A tuple containing:
                - SampleData object
                - Fixed/Flexible grid representation of the sample of shape (T, E, M)
                - Stats of the sample
        """
        sample = self.data_stats[ind]

        if self.feature_type == "fixed":
            grid, _ = sample.feature.to_fixed_grid(steps_per_quarter=self.steps_per_quarter)
            return sample, grid
        elif self.feature_type == "flexible":
            grid, _ = sample.feature.to_flexible_grid(max_hits_per_class=self.data_stats.max_hits_per_class, steps_per_quarter=self.steps_per_quarter)
            return sample, grid
        
    
    
    def collate_fn(self, batch: List[Tuple[SampleData, torch.Tensor, Dict]]) -> Dict[str, torch.Tensor]:
        """
        Collate a batch of (SampleData, grid, stats) into padded tensors and metadata.
        
        Returns:
            Dict with:
                - 'grid': Tensor of shape (B, T_max, E, M)
                - 'samples': List of SampleData
                - 'stats': List of stats
        """
        samples, grids = zip(*batch)
        T_max = max(g.shape[0] for g in grids)
        E, M = grids[0].shape[1:]

        padded_grids = [
            torch.cat([g, torch.zeros((T_max - g.shape[0], E, M))], dim=0)
            if g.shape[0] < T_max else g
            for g in grids
        ]

        return {
            'grid': torch.stack(padded_grids),  # (B, T_max, E, M)
            'samples': list(samples)
        }


    def query(self, **kwargs) -> List[SampleData]:
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
        return self.data_stats.query(**kwargs)


    def filter(self, **kwargs) -> None:
        """
        Filter the dataset with the given filters. 
        It will update the data_stats object and the length of the dataset.
        Args:
            styles: List of styles to filter by.
            time_signatures: List of time signatures to filter by.
            types: List of sample types to filter by.
            num_bars: Either a list of exact bar counts or a tuple (min_bars, max_bars) for a range query.
        
        Returns:
            None
        """
        self.data_stats = self.data_stats.filter(**kwargs)
        self.length = len(self.data_stats)
    
        

if __name__ == "__main__":
    import os
    import numpy as np
    from torch.utils.data import DataLoader

    # Setup path
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(project_root, "dataset", "serialized", "merged.pkl")

    # Test 1: Load dataset and single item
    dataset = DrumMIDIDataset(path=data_path, num_bars=2, feature_type="fixed", steps_per_quarter=4, subset=0.1, verbose=True)

    # Reconstruction test
    num_tests = 10
    for i in range(num_tests):
        rand_idx = np.random.randint(len(dataset))
        sample, grid = dataset[rand_idx]
        print(f"✅ Loaded one sample: grid.shape = {grid.shape}")
        reconstructed_feature = sample.feature.from_fixed_grid(grid, steps_per_quarter=dataset.steps_per_quarter)
        sample.feature.play()
        reconstructed_feature.play()

    # Test 2: Try collate_fn with DataLoader
    def run_collate_test(feature_type: str, steps_per_quarter: int):
        print(f"\n🔎 Testing feature_type='{feature_type}', steps_per_quarter={steps_per_quarter}")
        dataset = DrumMIDIDataset(path=data_path, num_bars=2, feature_type=feature_type, steps_per_quarter=steps_per_quarter, subset=0.1)
        loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=dataset.collate_fn)
        batch = next(iter(loader))

        grid = batch["grid"]
        samples = batch["samples"]

        print(f"✅ Collate successful: grid.shape = {grid.shape}")
        assert grid.ndim == 4, "Expected grid of shape (B, T, E, M)"
        assert len(samples) == grid.shape[0], "Mismatch between grid and samples"

        if feature_type == "fixed":
            for i, s in enumerate(samples):
                g, _ = s.feature.to_fixed_grid(steps_per_quarter)
                if g.shape[0] < grid.shape[1]:
                    pad = grid[i, g.shape[0]:, :, :]
                    assert torch.all(pad == 0), f"Non-zero padding at sample {i}"
            print("✅ Padding check passed.")

    # Run both fixed and flexible tests
    for ft in ["fixed", "flexible"]:
        for spq in [2, 4]:
            run_collate_test(feature_type=ft, steps_per_quarter=spq)
    


    
