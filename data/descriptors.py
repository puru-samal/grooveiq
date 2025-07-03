import numpy as np
'''
from scipy import stats
from scipy.signal import find_peaks
from scipy.spatial.distance import cdist
from scipy.stats import gaussian_kde
from scipy.integrate import simps
import matplotlib.pyplot as plt
'''
from .GrooveToolbox import NewGroove
from data import DrumMIDIFeature
import torch
from typing import Tuple, List

# =========================
# FEATURE DESCRIPTORS CLASS
# =========================

class FeatureDescriptors:
    SIXTEENTH_NOTES_PER_BAR = 16
    MAX_BARS = 2
    MATRIX_SIZE = SIXTEENTH_NOTES_PER_BAR * MAX_BARS
    NUM_DRUM_PIECES = 10
    
    drum_mapping = {
        "kick": 0,
        "snare": 1,
        "hh_closed": 2,
        "hh_open": 3,
        "ride": 4,
        "crash": 5,
        "extra_cymbal": 6, # Dummy piece to be compatible with GrooveToolbox
        "low_tom": 7,
        "mid_tom": 8,
        "high_tom": 9
    }

    def __init__(self, feature : DrumMIDIFeature, velocity_type: str = "regular"):
        self.feature = feature
        self.pm_obj = feature.to_pretty_midi()
        tempo = feature.score.tempos[0].tempo
        hits = self._get_all_hit_info(self.pm_obj, tempo, feature.canonical_map)

        hits_matrix = np.zeros([32, 10])  # todo: work with loops of arbritrary length
        timing_matrix = np.zeros([32, 10])
        timing_matrix[:] = np.nan

        for j in range(hits.shape[0]):
            time_position = int(hits[j, 0] * 4)
            kit_piece_position = int(hits[j, 2])
            timing_matrix[time_position % 32, kit_piece_position] = hits[j, 3]
            hits_matrix[time_position % 32, kit_piece_position] = hits[j, 1]

        hits_matrix = hits_matrix
        timing_matrix = timing_matrix
        tempo = tempo

        self.groove = NewGroove(hits_matrix, timing_matrix, tempo, extract_features=True, velocity_type=velocity_type)
        self.descriptors = {
            "complexity" : self.groove.RhythmFeatures.total_complexity,
            "lowness"    : self.groove.RhythmFeatures.low_density,
            "midness"    : self.groove.RhythmFeatures.mid_density,
            "highness"   : self.groove.RhythmFeatures.high_density
        }

    def _get_all_hit_info(self, midi, tempo, keymap):
    # Create an array of all hits in a groove
    # Format: [quantized time index, velocity, kit piece, microtiming deviation from metronome]

        hits = np.zeros([len(midi.instruments[0].notes), 4])
        i = 0
        for instrument in midi.instruments:
            for note in instrument.notes:
                hits[i, 0] = note.start
                hits[i, 1] = note.velocity
                hits[i, 2] = self.drum_mapping[keymap[note.pitch]]
                i += 1

        hits[:, 0] = hits[:, 0] / 60.0 * tempo
        hits[:, 1] = hits[:, 1] / 127.0
        multiplied_hit = hits[:, 0] * 4.0
        rounded_hits = multiplied_hit.round(decimals=0) / 4.0
        microtiming_variation_Beats = hits[:, 0] - rounded_hits
        microtiming_variation_MS = microtiming_variation_Beats * 60.0 * 1000 / tempo
        hits[:, 0] = rounded_hits
        hits[:, 3] = microtiming_variation_MS
        return hits

    def get_feature_vector(self) -> Tuple[torch.Tensor, List[str]]:
        keys = list(self.descriptors.keys())
        vector = torch.tensor([self.descriptors[k] for k in keys])
        return vector, keys

    def to_dict(self):
        return self.descriptors

    def print_all_features(self):
        for k, v in self.descriptors.items():
            print(f"{k}: {v:.4f}")


if __name__ == "__main__":
    import pickle
    import numpy as np
    from data import DataStats

    path = "/Users/puruboii/Desktop/Github-Local/drum-genie/dataset/serialized/merged_ts=4-4_2bar_tr0.80-va0.10-te0.10_test.pkl"
    name = path.split('/')[-1].split('.')[0]
    with open(path, "rb") as f:
        dataset = pickle.load(f)

    test_set = DataStats()
    test_set.set_name(name = name)

    for datapoint in dataset:
        test_set.accumulate_dict(datapoint)

    random_idx = np.random.randint(0, len(dataset))
    sample = test_set.all_samples[random_idx]
    feature = sample.feature
    feature.play()
    descriptor = sample.descriptors
    vector, keys = descriptor.get_feature_vector()
    print(vector)
    print(keys)


'''
# =========================
# ANALYSIS + VISUALIZATION
# =========================

def compute_distance_matrix(vectors):
    return cdist(vectors, vectors, metric="euclidean")

def compute_pdf(distances):
    kde = gaussian_kde(distances, bw_method="scott")
    x_grid = np.linspace(np.min(distances), np.max(distances), 200)
    pdf_values = kde(x_grid)
    return x_grid, pdf_values

def kl_divergence(p, q):
    p = p / (np.sum(p) + 1e-10)
    q = q / (np.sum(q) + 1e-10)
    return np.sum(p * np.log((p + 1e-10) / (q + 1e-10)))

def overlap_area(p, q):
    return simps(np.minimum(p, q)) / simps(p)

def analyze_features(ref_vectors, gen_vectors):
    intra_distances = compute_distance_matrix(ref_vectors)
    intra_flat = intra_distances[np.triu_indices_from(intra_distances, k=1)].flatten()

    inter_distances = cdist(gen_vectors, ref_vectors)
    inter_flat = inter_distances.flatten()

    x_intra, pdf_intra = compute_pdf(intra_flat)
    x_inter, pdf_inter = compute_pdf(inter_flat)

    kl = kl_divergence(pdf_inter, pdf_intra)
    oa = overlap_area(pdf_inter, pdf_intra)

    return {
        "KL": kl,
        "OA": oa,
        "x_intra": x_intra,
        "pdf_intra": pdf_intra,
        "x_inter": x_inter,
        "pdf_inter": pdf_inter
    }

def plot_pdf_comparison(x_intra, pdf_intra, x_inter, pdf_inter, title="PDF Comparison"):
    plt.figure(figsize=(8, 5))
    plt.plot(x_intra, pdf_intra, label="Intra-set (reference)", linewidth=2)
    plt.plot(x_inter, pdf_inter, label="Inter-set (generated)", linewidth=2)
    plt.xlabel("Distance")
    plt.ylabel("Density")
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.show()

'''