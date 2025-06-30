from .feature import DrumMIDIFeature
import numpy as np
from typing import Literal
import math
import matplotlib.pyplot as plt
from scipy import stats
from scipy.signal import find_peaks
from pandas.plotting import autocorrelation_plot


'''
A collection of descriptors/metrics for drum loops. 
Sources: 
https://github.com/fredbru/GrooveToolbox

A groove is described here as a short drum loop of arbritary length and polyphony

Vector format:
0 Kick
1 Snare
2 Closed Hihat
3 Open Hihat
4 Ride
5 Crash
6 Low tom
7 Mid tom
8 High tom

'''

SIXTEENTH_NOTES_PER_BAR = 16  # 4 beats * 4 sixteenth notes per beat
MAX_BARS = 2  # Maximum number of bars to process
MATRIX_SIZE = SIXTEENTH_NOTES_PER_BAR * MAX_BARS  # 32 time positions
NUM_DRUM_PIECES = 9  # Number of different drum pieces

class FeatureDescriptors:
    drum_mapping = {
        "kick": 0,
        "snare": 1, 
        "hh_closed": 2,
        "hh_open": 3,
        "ride": 4,
        "crash": 5,
        "low_tom": 6,
        "mid_tom": 7,
        "high_tom": 8
    }

    '''
    velocity_type:
    - regular: use the velocity of the note
    - none: use the velocity of the note
    - transform: transform the velocity of the note

    Descriptors Available:
    - RhythmFeatures:
        - combined_syncopation
        - polyphonic_syncopation
        - low_syncopation
        - mid_syncopation
        - high_syncopation
        - low_density
        - mid_density
        - high_density
        - total_density
        - hiness
        - hisyncness
        - autocorrelation_skew
        - autocorrelation_max_amplitude
        - autocorrelation_centroid
        - autocorrelation_harmonicity
        - total_symmetry
        - total_average_intensity
        - total_complexity

    - MicrotimingFeatures:
        - swingness
        - is_swung
        - laidbackness
        - timing_accuracy
    '''
    
    def __init__(self, feature: DrumMIDIFeature, velocity_type : Literal["regular", "none", "transform"] = "regular"):
        self.pm_obj = feature.to_pretty_midi()
        self.tempo  = feature.score.tempos[0].tempo
        hits = self._get_all_hit_info(self.pm_obj, self.tempo, feature.canonical_map)
        
        # Initialize matrices
        self.groove_9_parts = np.zeros([MATRIX_SIZE, NUM_DRUM_PIECES])  
        self.timing_matrix = np.zeros([MATRIX_SIZE, NUM_DRUM_PIECES])
        self.timing_matrix[:] = np.nan

        # Populate matrices
        for j in range(hits.shape[0]):
            time_position = int(hits[j, 0] * 4)
            kit_piece_position = int(hits[j, 2])
            self.timing_matrix[time_position % MATRIX_SIZE, kit_piece_position] = hits[j, 3]
            self.groove_9_parts[time_position % MATRIX_SIZE, kit_piece_position] = hits[j, 1]

        if velocity_type not in ["regular", "none", "transform"]:
            raise ValueError(f"Invalid velocity type: {velocity_type}")
        if velocity_type == "none":
            self.groove_9_parts = np.ceil(self.groove_9_parts)
        elif velocity_type == "regular":
            pass
        elif velocity_type == "transform":
            self.groove_9_parts = np.power(self.groove_9_parts, 0.2)

        self.groove_5_parts = self._groupGroove5KitParts()
        self.groove_3_parts = self._groupGroove3KitParts()
        self.RhythmFeatures = RhythmFeatures(self.groove_9_parts, self.groove_5_parts, self.groove_3_parts)
        self.MicrotimingFeatures = MicrotimingFeatures(self.timing_matrix, self.tempo)
        self.RhythmFeatures.calculate_all_features()
        self.MicrotimingFeatures.calculate_all_features()

        self.descriptors = {}
        self.descriptors.update(self.RhythmFeatures.get_all_features())
        self.descriptors.update(self.MicrotimingFeatures.get_all_features())

    @classmethod


    def get_all_features(self):
        pass

    def print_all_features(self):
        pass

    def calculate_noi(self):
        "Total number of instruments used in a sample"
        piano_roll = self.pm_obj.instruments[0].get_piano_roll(fs=100)
        sum_notes = np.sum(piano_roll, axis=1)
        used_instruments = np.sum(sum_notes > 0)
        return used_instruments
    
    def calculate_transition_matrix(self):
        pass

    def calculate_avg_ioi(self):
        pass

    def _groupGroove5KitParts(self):
        """
        Group 9 drum kit parts into 5 polyphony levels.
        
        Groups:
        0 - Kick
        1 - Snare  
        2 - Closed cymbals (closed hihat + ride)
        3 - Open cymbals (open hihat + crash)
        4 - Toms (low + mid + high tom)
        
        Returns:
            numpy.ndarray: Matrix of shape (32, 5) with grouped drum parts
        """
        # Validate input matrix exists and has correct shape
        if not hasattr(self, 'groove_9_parts') or self.groove_9_parts.shape[1] != 9:
            raise ValueError("groove_9_parts matrix must exist with 9 columns")
        
        # Define drum piece indices
        Group_1 = [0] # Kick
        Group_2 = [1] # Snare
        Group_3 = [2, 4] # Closed cymbals
        Group_4 = [3, 5] # Open cymbals
        Group_5 = [6, 7, 8] # Toms
        
        # Initialize output matrix
        groove_5_parts = np.zeros([self.groove_9_parts.shape[0], 5])
        
        # Group 1: Kick (unchanged)
        groove_5_parts[:, 0] = self.groove_9_parts[:, Group_1].sum(axis=1)
        
        # Group 2: Snare (unchanged)
        groove_5_parts[:, 1] = self.groove_9_parts[:, Group_2].sum(axis=1)
        
        # Group 3: Closed cymbals (closed hihat + ride)
        closed_cymbals = self.groove_9_parts[:, Group_3].sum(axis=1)
        groove_5_parts[:, 2] = np.clip(closed_cymbals, 0, 1)
        
        # Group 4: Open cymbals (open hihat + crash)
        open_cymbals = self.groove_9_parts[:, Group_4].sum(axis=1)
        groove_5_parts[:, 3] = np.clip(open_cymbals, 0, 1)
        
        # Group 5: Toms (low + mid + high tom)
        toms = self.groove_9_parts[:, Group_5].sum(axis=1)
        groove_5_parts[:, 4] = np.clip(toms, 0, 1)
        
        return groove_5_parts

    def _groupGroove3KitParts(self):
        """
        Group 5 drum kit parts into 3 frequency ranges.
        
        Groups:
        0 - Low frequency (kick)
        1 - Mid frequency (snare + toms)
        2 - High frequency (closed cymbals + open cymbals)
        
        Returns:
            numpy.ndarray: Matrix of shape (32, 3) with grouped drum parts
        """
        # Validate input matrix exists and has correct shape
        if not hasattr(self, 'groove_5_parts') or self.groove_5_parts.shape[1] != 5:
            raise ValueError("groove_5_parts matrix must exist with 5 columns")
        
        # Define drum piece indices for clarity
        Group_1 = [0] # Kick
        Group_2 = [1, 4] # Snare + Toms
        Group_3 = [2, 3] # Closed cymbals + Open cymbals
        
        # Initialize output matrix
        groove_3_parts = np.zeros([self.groove_5_parts.shape[0], 3])
        
        # Group 1: Low frequency - Kick (unchanged)
        groove_3_parts[:, 0] = self.groove_5_parts[:, Group_1].sum(axis=1)
        
        # Group 2: Mid frequency - Snare + Toms
        mid_frequency = self.groove_5_parts[:, Group_2].sum(axis=1)
        groove_3_parts[:, 1] = np.clip(mid_frequency, 0, 1)
        
        # Group 3: High frequency - Closed cymbals + Open cymbals
        high_frequency = self.groove_5_parts[:, Group_3].sum(axis=1)
        groove_3_parts[:, 2] = np.clip(high_frequency, 0, 1)
        
        return groove_3_parts

    def reduce_groove(self):
        """
        Remove ornamentation from a groove to return a simplified representation of the rhythm structure.
        
        This function applies metrical reduction to simplify complex rhythms by:
        1. Removing weak hits (velocity <= 0.4)
        2. Applying density and figural transforms based on metrical strength
        3. Handling syncopation by shifting notes to stronger metrical positions
        
        Returns:
            numpy.ndarray: Reduced groove matrix with simplified rhythm structure
        """
        # Validate input matrix exists
        if not hasattr(self, 'groove_9_parts') or self.groove_9_parts.shape[1] != 9:
            raise ValueError("groove_9_parts matrix must exist with 9 columns")
        
        # Metrical profile for 4/4 time with 16th note resolution (32 positions)
        # Values represent metrical strength: 0 = strongest, -2 = weakest
        metrical_profile = [
            0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2,  # Bar 1
            0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2, 0, -2, -1, -2   # Bar 2
        ]
        
        # Initialize reduced groove matrix
        self.reduced_groove = np.zeros(self.groove_9_parts.shape)
        
        # Apply reduction to each drum part
        for part_index in range(9):  # 9 drum parts
            self.reduced_groove[:, part_index] = self._reduce_part(
                self.groove_9_parts[:, part_index].copy(), 
                metrical_profile
            )
        
        # Remove off-beat positions to get downbeat-only version
        # Keep only positions 0, 4, 8, 12, 16, 20, 24, 28 (every 4th position)
        rows_to_remove = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 
                          17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
        self.reduced_groove = np.delete(self.reduced_groove, rows_to_remove, axis=0)
        
        return self.reduced_groove

    def _reduce_part(self, part, metrical_profile):
        """
        Apply metrical reduction to a single drum part.
        
        Args:
            part (numpy.ndarray): 1D array representing a single drum part
            metrical_profile (list): List of metrical strength values
            
        Returns:
            numpy.ndarray: Reduced drum part
        """
        # Constants for better readability
        VELOCITY_THRESHOLD = 0.4  # Minimum velocity to consider a hit significant
        LOOKBACK_RANGE = 3  # Number of previous positions to check
        
        length = len(part)
        
        # Step 1: Remove weak hits (velocity <= threshold)
        part[part <= VELOCITY_THRESHOLD] = 0
        
        # Step 2: Apply density and figural transforms
        for i in range(length):
            if part[i] != 0:  # Hit detected at position i
                # Check previous positions for weaker metrical events
                for k in range(max(0, i - LOOKBACK_RANGE), i):
                    if part[k] != 0 and metrical_profile[k] < metrical_profile[i]:
                        # Found a preceding event in a weaker metrical position
                        
                        # Find the strongest metrical position between previous event and k
                        previous_event_index = self._find_previous_event(part, k)
                        strongest_position = self._find_strongest_position(
                            metrical_profile, previous_event_index, k
                        )
                        
                        # Apply transform based on metrical strength
                        if strongest_position <= k:
                            # Density transform: remove the note
                            part[k] = 0
                        else:
                            # Figural transform: shift note to stronger position
                            part[strongest_position] = part[k]
                            part[k] = 0
        
        # Step 3: Handle syncopation
        for i in range(length):
            if part[i] == 0:  # Empty position
                # Check if there's a syncopated note that should be shifted here
                for k in range(max(0, i - LOOKBACK_RANGE), i):
                    if part[k] != 0 and metrical_profile[k] < metrical_profile[i]:
                        # Syncopation detected: shift note from weak to strong position
                        part[i] = part[k]
                        part[k] = 0
        
        return part

    def _find_previous_event(self, part, position):
        """
        Find the index of the previous event before the given position.
        
        Args:
            part (numpy.ndarray): Drum part array
            position (int): Current position
            
        Returns:
            int: Index of previous event, or 0 if none found
        """
        for i in range(position - 1, -1, -1):
            if part[i] != 0:
                return i
        return 0

    def _find_strongest_position(self, metrical_profile, start, end):
        """
        Find the position with the strongest metrical value in the given range.
        
        Args:
            metrical_profile (list): Metrical strength values
            start (int): Start position (inclusive)
            end (int): End position (exclusive)
            
        Returns:
            int: Position with strongest metrical value
        """
        if start >= end:
            return start
        
        # Find the maximum metrical value in the range
        max_strength = max(metrical_profile[start:end])
        
        # Return the first position with this maximum value
        for i in range(start, end):
            if metrical_profile[i] == max_strength:
                return i
        
        return start

    def _get_all_hit_info(self, midi, tempo, key_map):
        """
        Create an array of all hits in a groove.
        Args:
            midi (pretty_midi.PrettyMIDI): MIDI object
            tempo (float): Tempo of the MIDI file
            key_map (dict): Mapping of pitch to drum piece name
        
        Returns:
            numpy.ndarray: Array with format [quantized_time, velocity, kit_piece, microtiming_deviation_ms]
        """
        # Initialize hits array: [time, velocity, kit_piece, microtiming_deviation]
        total_notes = sum(len(instrument.notes) for instrument in midi.instruments)
        hits = np.zeros([total_notes, 4])
        
        hit_index = 0
        for instrument in midi.instruments:
            for note in instrument.notes:
                # Store raw timing and velocity
                hits[hit_index, 0] = note.start  # Time in seconds
                hits[hit_index, 1] = note.velocity  # Velocity (0-127)
                
                # Map pitch to drum kit piece
                hits[hit_index, 2] = self.drum_mapping[key_map[note.pitch]]
                hit_index += 1
        
        # Convert timing to beats and normalize velocity
        hits[:, 0] = hits[:, 0] / 60.0 * tempo  # Convert seconds to beats
        hits[:, 1] = hits[:, 1] / 127.0  # Normalize velocity to 0-1
        
        # Quantize timing to 16th notes (4 beats per bar)
        # Multiply by 4 to get 16th note resolution, round, then divide by 4
        sixteenth_note_times = hits[:, 0] * 4.0
        quantized_beats = sixteenth_note_times.round(decimals=0) / 4.0
        
        # Calculate microtiming deviation
        microtiming_deviation_beats = hits[:, 0] - quantized_beats
        microtiming_deviation_ms = microtiming_deviation_beats * 60.0 * 1000 / tempo
        
        # Update hits array with quantized timing and microtiming deviation
        hits[:, 0] = quantized_beats
        hits[:, 3] = microtiming_deviation_ms
        
        return hits
    
    def get_features(self):
        return self.feature.get_features()
    
    def weighted_hamming_distance(self, other_groove, parts_count=9, beat_weighting=False):
        """
        Calculate weighted Hamming distance between two grooves.
        
        Args:
            other_groove (GrooveToolboxFeatures): Another groove to compare with
            parts_count (int): Number of drum parts to use (3, 5, or 9)
            beat_weighting (str): Whether to apply metrical weighting ("On" or "Off")
            
        Returns:
            float: Weighted Hamming distance
        """
        if parts_count == 3:
            a = self.groove_3_parts
            b = other_groove.groove_3_parts
        elif parts_count == 5:
            a = self.groove_5_parts
            b = other_groove.groove_5_parts
        elif parts_count == 9:
            a = self.groove_9_parts
            b = other_groove.groove_9_parts
        else:
            raise ValueError("parts_count must be 3, 5, or 9")

        if beat_weighting:
            a = self._weight_groove(a)
            b = self._weight_groove(b)

        x = (a.flatten() - b.flatten())
        return math.sqrt(np.dot(x, x.T))

    def fuzzy_hamming_distance(self, other_groove, parts_count=9, beat_weighting="Off"):
        """
        Calculate fuzzy Hamming distance with velocity weighting and microtiming consideration.
        
        Args:
            other_groove (GrooveToolboxFeatures): Another groove to compare with
            parts_count (int): Number of drum parts to use (3, 5, or 9)
            beat_weighting (str): Whether to apply metrical weighting ("On" or "Off")
            
        Returns:
            float: Fuzzy Hamming distance
        """
        if parts_count == 3:
            a = self.groove_3_parts
            b = other_groove.groove_3_parts
            a_timing = None
            b_timing = None
        elif parts_count == 5:
            a = self.groove_5_parts
            b = other_groove.groove_5_parts
            a_timing = None
            b_timing = None
        elif parts_count == 9:
            a = self.groove_9_parts
            b = other_groove.groove_9_parts
            a_timing = self.timing_matrix
            b_timing = other_groove.timing_matrix
        else:
            raise ValueError("parts_count must be 3, 5, or 9")

        if beat_weighting == "On":
            a = self._weight_groove(a)
            b = self._weight_groove(b)

        # For 3 and 5 parts, use simple Hamming distance
        if a_timing is None:
            x = (a.flatten() - b.flatten())
            return math.sqrt(np.dot(x, x.T))

        # For 9 parts, use fuzzy distance with timing
        timing_difference = np.nan_to_num(a_timing - b_timing)

        x = np.zeros(a.shape)
        tempo = self.tempo
        steptime_ms = 60.0 * 1000 / tempo / 4  # semiquaver step time in ms

        difference_weight = timing_difference / 125.
        difference_weight = 1 + np.absolute(difference_weight)
        single_difference_weight = 400

        for j in range(parts_count):
            for i in range(31):
                if a[i, j] != 0.0 and b[i, j] != 0.0:
                    x[i, j] = (a[i, j] - b[i, j]) * (difference_weight[i, j])
                elif a[i, j] != 0.0 and b[i, j] == 0.0:
                    if b[(i + 1) % 32, j] != 0.0 and a[(i + 1) % 32, j] == 0.0:
                        single_difference = np.nan_to_num(a_timing[i, j]) - np.nan_to_num(b_timing[(i + 1) % 32, j]) + steptime_ms
                        if single_difference < 125.:
                            single_difference_weight = 1 + abs(single_difference_weight / steptime_ms)
                            x[i, j] = (a[i, j] - b[(i + 1) % 32, j]) * single_difference_weight
                        else:
                            x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                    elif b[(i - 1) % 32, j] != 0.0 and a[(i - 1) % 32, j] == 0.0:
                        single_difference = np.nan_to_num(a_timing[i, j]) - np.nan_to_num(b_timing[(i - 1) % 32, j]) - steptime_ms

                        if single_difference > -125.:
                            single_difference_weight = 1 + abs(single_difference_weight / steptime_ms)
                            x[i, j] = (a[i, j] - b[(i - 1) % 32, j]) * single_difference_weight
                        else:
                            x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]
                    else:
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                elif a[i, j] == 0.0 and b[i, j] != 0.0:
                    if b[(i + 1) % 32, j] != 0.0 and a[(i + 1) % 32, j] == 0.0:
                        single_difference = np.nan_to_num(a_timing[i, j]) - np.nan_to_num(b_timing[(i + 1) % 32, j]) + steptime_ms
                        if single_difference < 125.:
                            single_difference_weight = 1 + abs(single_difference_weight / steptime_ms)
                            x[i, j] = (a[i, j] - b[(i + 1) % 32, j]) * single_difference_weight
                        else:
                            x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                    elif b[(i - 1) % 32, j] != 0.0 and a[(i - 1) % 32, j] == 0.0:
                        single_difference = np.nan_to_num(a_timing[i, j]) - np.nan_to_num(b_timing[(i - 1) % 32, j]) - steptime_ms
                        if single_difference > -125.:
                            single_difference_weight = 1 + abs(single_difference_weight / steptime_ms)
                            x[i, j] = (a[i, j] - b[(i - 1) % 32, j]) * single_difference_weight

                        else:
                            x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

                    else:  # if no nearby onsets, need to count difference between onset and 0 value.
                        x[i, j] = (a[i, j] - b[i, j]) * difference_weight[i, j]

        fuzzy_distance = math.sqrt(np.dot(x.flatten(), x.flatten().T))
        return fuzzy_distance

    def structural_similarity_distance(self, other_groove):
        """
        Calculate structural similarity between reduced versions of two grooves.
        
        Args:
            other_groove (GrooveToolboxFeatures): Another groove to compare with
            
        Returns:
            float: Structural similarity distance
        """
        a = self.reduce_groove()
        b = other_groove.reduce_groove()
        
        # Remove off-beat positions to get downbeat-only version
        rows_to_remove = [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 
                          17, 18, 19, 21, 22, 23, 25, 26, 27, 29, 30, 31]
        reduced_a = np.delete(a, rows_to_remove, axis=0)
        reduced_b = np.delete(b, rows_to_remove, axis=0)
        
        x = (reduced_a.flatten() - reduced_b.flatten())
        structural_difference = math.sqrt(np.dot(x, x.T))
        return structural_difference

    def _weight_groove(self, groove):
        """
        Apply metrical awareness profile weighting for distance calculations.
        
        Args:
            groove (numpy.ndarray): Groove matrix to weight
            
        Returns:
            numpy.ndarray: Weighted groove matrix
        """
        # Metrical awareness profile weighting for hamming distance.
        # The rhythms in each beat of a bar have different significance based on GTTM.
        beat_awareness_weighting = [1, 1, 1, 1,
                                   0.27, 0.27, 0.27, 0.27,
                                   0.22, 0.22, 0.22, 0.22,
                                   0.16, 0.16, 0.16, 0.16,
                                   1, 1, 1, 1,
                                   0.27, 0.27, 0.27, 0.27,
                                   0.22, 0.22, 0.22, 0.22,
                                   0.16, 0.16, 0.16, 0.16]

        weighted_groove = groove.copy()
        for i in range(groove.shape[1]):
            weighted_groove[:, i] = groove[:, i] * beat_awareness_weighting
        return weighted_groove

class RhythmFeatures():
    def __init__(self, groove_9_parts, groove_5_parts, groove_3_parts):
        self.groove_9_parts = groove_9_parts
        self.groove_5_parts = groove_5_parts
        self.groove_3_parts = groove_3_parts
        self.total_autocorrelation_curve = None

    def calculate_all_features(self):
        # Get all standard features in one go
        self.combined_syncopation = self.get_combined_syncopation()
        self.polyphonic_syncopation = self.get_polyphonic_syncopation()
        self.low_syncopation = self.get_low_syncopation()
        self.mid_syncopation = self.get_mid_syncopation()
        self.high_syncopation = self.get_high_syncopation()
        self.low_density = self.get_low_density()
        self.mid_density = self.get_mid_density()
        self.high_density = self.get_high_density()
        self.total_density = self.get_total_density()
        self.hiness = self.get_hiness()
        self.hisyncness = self.get_hisyncness()
        self.autocorrelation_skew = self.get_autocorrelation_skew()
        self.autocorrelation_max_amplitude = self.get_autocorrelation_max_amplitude()
        self.autocorrelation_centroid = self.get_autocorrelation_centroid()
        self.autocorrelation_harmonicity = self.get_autocorrelation_harmonicity()
        self.total_symmetry = self.get_total_symmetry()
        self.total_average_intensity = self.get_total_average_intensity()
        self.total_weak_to_strong_ratio = self.get_total_weak_to_strong_ratio()
        self.total_complexity = self.get_total_complexity()

    def get_all_features(self) -> dict:
        self.calculate_all_features()
        return {
            "combined_syncopation": self.combined_syncopation,
            "polyphonic_syncopation": self.polyphonic_syncopation,
            "low_syncopation": self.low_syncopation,
            "mid_syncopation": self.mid_syncopation,
            "high_syncopation": self.high_syncopation,
            "low_density": self.low_density,
            "mid_density": self.mid_density,
            "high_density": self.high_density,
            "total_density": self.total_density,
            "hiness": self.hiness,
            "hisyncness": self.hisyncness,
            "autocorrelation_skew": self.autocorrelation_skew,
            "autocorrelation_max_amplitude": self.autocorrelation_max_amplitude,
            "autocorrelation_centroid": self.autocorrelation_centroid,
            "autocorrelation_harmonicity": self.autocorrelation_harmonicity,
            "total_symmetry": self.total_symmetry,
            "total_average_intensity": self.total_average_intensity,
            "total_weak_to_strong_ratio": self.total_weak_to_strong_ratio,
            "total_complexity": self.total_complexity,
        }

    def get_combined_syncopation(self):
        # Calculate syncopation as summed across all kit parts.
        self.combined_syncopation = 0.0
        for i in range(self.groove_9_parts.shape[1]):
            self.combined_syncopation += self.get_syncopation_1part(self.groove_9_parts[:,i])
        return self.combined_syncopation

    def get_polyphonic_syncopation(self):
        # Calculate syncopation using Witek syncopation distance - modelling syncopation between instruments
        # Works on semiquaver and quaver levels of syncopation

        metrical_profile = [0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3,
                           0, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3, -1, -3, -2, -3]

        low = self.groove_3_parts[:,0]
        mid = self.groove_3_parts[:,1]
        high = self.groove_3_parts[:,2]

        total_syncopation = 0

        for i in range(len(low)):
            kick_syncopation = self._get_kick_syncopation(low, mid, high, i, metrical_profile)
            snare_syncopation = self._get_snare_syncopation(low, mid, high, i, metrical_profile)
            total_syncopation += kick_syncopation * low[i]
            total_syncopation += snare_syncopation * mid[i]
        return total_syncopation

    def _get_kick_syncopation(self, low, mid, high, i, metrical_profile):
        # Find instances  when kick syncopates against hi hat/snare on the beat.
        # For use in polyphonic syncopation feature

        kick_syncopation = 0
        k = 0
        next_hit = ""
        if low[i] == 1 and low[(i + 1) % 32] !=1 and low[(i+2) % 32] != 1:
            for j in i + 1, i + 2, i + 3, i + 4: #look one and two steps ahead only - account for semiquaver and quaver sync
                if mid[(j % 32)] == 1 and high[(j % 32)] != 1:
                    next_hit = "Mid"
                    k = j % 32
                    break
                elif high[(j % 32)] == 1 and mid[(j % 32)] != 1:
                    next_hit = "High"
                    k = j % 32
                    break
                elif high[(j % 32)] == 1 and mid[(j % 32)] == 1:
                    next_hit = "MidAndHigh"
                    k = j % 32
                    break
                # if both next two are 0 - next hit == rest. get level of the higher level rest
            if mid[(i+1)%32] + mid[(i+2)%32] == 0.0 and high[(i+1)%32] + [(i+2)%32] == 0.0:
                next_hit = "None"

            if next_hit == "MidAndHigh":
                if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                    difference = metrical_profile[k] - metrical_profile[i]
                    kick_syncopation = difference + 2
            elif next_hit == "Mid":
                if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                    difference = metrical_profile[k] - metrical_profile[i]
                    kick_syncopation = difference + 2
            elif next_hit == "High":
                if metrical_profile[k] >= metrical_profile[i]:
                    difference = metrical_profile[k] - metrical_profile[i]
                    kick_syncopation = difference + 5
            elif next_hit == "None":
                if metrical_profile[k] > metrical_profile[i]:
                    difference = max(metrical_profile[(i + 1) % 32], metrical_profile[(i + 2) % 32]) - metrical_profile[i]
                    kick_syncopation = difference + 6 # if rest on a stronger beat - one stream sync, high sync valuef
        return kick_syncopation

    def _get_snare_syncopation(self, low, mid, high, i, metrical_profile):
        # Find instances  when snare syncopates against hi hat/kick on the beat
        # For use in polyphonic syncopation feature

        snare_syncopation = 0
        next_hit = ""
        k = 0
        if mid[i] == 1 and mid[(i + 1) % 32] !=1 and mid[(i+2) % 32] != 1:
            for j in i + 1, i + 2, i + 3, i + 4: #look one and 2 steps ahead only
                if low[(j % 32)] == 1 and high[(j % 32)] != 1:
                    next_hit = "Low"
                    k = j % 32
                    break
                elif high[(j % 32)] == 1 and low[(j % 32)] != 1:
                    next_hit = "High"
                    k = j % 32
                    break
                elif high[(j % 32)] == 1 and low[(j % 32)] == 1:
                    next_hit = "LowAndHigh"
                    k = j % 32
                    break
            if low[(i+1)%32] + low[(i+2)%32] == 0.0 and high[(i+1)%32] + [(i+2)%32] == 0.0:
                next_hit = "None"

            if next_hit == "LowAndHigh":
                if metrical_profile[k] >= metrical_profile[i]:
                    difference = metrical_profile[k] - metrical_profile[i]
                    snare_syncopation = difference + 1  # may need to make this back to 1?)
            elif next_hit == "Low":
                if metrical_profile[k] >= metrical_profile[i]:
                    difference = metrical_profile[k] - metrical_profile[i]
                    snare_syncopation = difference + 1
            elif next_hit == "High":
                if metrical_profile[k] >= metrical_profile[i]:  # if hi hat is on a stronger beat - syncopation
                    difference = metrical_profile[k] - metrical_profile[i]
                    snare_syncopation = difference + 5
            elif next_hit == "None":
                if metrical_profile[k] > metrical_profile[i]:
                    difference = max(metrical_profile[(i + 1) % 32], metrical_profile[(i + 2) % 32]) - metrical_profile[i]
                    snare_syncopation = difference + 6 # if rest on a stronger beat - one stream sync, high sync value
        return snare_syncopation

    def get_syncopation_1part(self, part):
        # Using Longuet-Higgins  and  Lee 1984 metric profile, get syncopation of 1 monophonic line.
        # Assumes it's a drum loop - loops round.
        # Normalized against maximum syncopation: syncopation score of pattern with all pulses of lowest metrical level
        # at maximum amplitude (=30 for 2 bar 4/4 loop)

        metrical_profile = [5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1,
                                   5, 1, 2, 1, 3, 1, 2, 1, 4, 1, 2, 1, 3, 1, 2, 1]
        max_syncopation = 30.0
        syncopation = 0.0
        for i in range(len(part)):
            if part[i] != 0:
                if part[(i + 1) % 32] == 0.0 and metrical_profile[(i + 1) % 32] > metrical_profile[i]:
                    syncopation = float(syncopation + (
                    abs(metrical_profile[(i + 1) % 32] - metrical_profile[i]))) # * part[i])) #todo: velocity here?

                elif part[(i + 2) % 32] == 0.0 and metrical_profile[(i + 2) % 32] > metrical_profile[i]:
                    syncopation = float(syncopation + (
                    abs(metrical_profile[(i + 2) % 32] - metrical_profile[i]))) # * part[i]))
        return syncopation / max_syncopation

    def get_average_intensity(self, part):
        # Get average loudness for any signle part or group of parts. Will return 1 for binary loop, otherwise calculate
        # based on velocity mode chosen (transform or regular)

        # first get all non-zero hits. then divide by number of hits

        hit_indexes = np.nonzero(part)
        total = 0.0
        hit_count = np.count_nonzero(part)

        for i in range(hit_count):
            if len(hit_indexes) > 1:
                index = hit_indexes[0][i], hit_indexes[1][i]
            else:
                index = hit_indexes[0][i]
            total += part[index]
        average = total / hit_count
        return average

    def get_total_average_intensity(self):
        # Get average loudness for every hit in a loop
        self.total_average_intensity = self.get_average_intensity(self.groove_9_parts)
        return self.total_average_intensity

    def get_weak_to_strong_ratio(self, part):
        weak_hit_count = 0.0
        strong_hit_count = 0.0

        strong_positions = [0,4,8,12,16,20,24,28]
        weak_positions = [1,2,3,5,6,7,9,10,11,13,14,15,17,18,19,21,22,23,25,26,27,29,30,31]

        hits_count = np.count_nonzero(part)
        hit_indexes = np.nonzero(part)
        for i in range(hits_count):
            if len(hit_indexes) > 1:
                index = hit_indexes[0][i], hit_indexes[1][i]
            else:
                index = [hit_indexes[0][i]]
            if index[0] in strong_positions:
                strong_hit_count += part[index]
            if index[0] in weak_positions:
                weak_hit_count += part[index]
        weakToStrongRatio = weak_hit_count/strong_hit_count
        return weakToStrongRatio

    def get_total_weak_to_strong_ratio(self): #todo: test
        self.total_weak_to_strong_ratio = self.get_weak_to_strong_ratio(self.groove_9_parts)
        return self.total_weak_to_strong_ratio

    def get_low_syncopation(self):
        # Get syncopation of low part (kick drum)
        self.low_syncopation = self.get_syncopation_1part(self.groove_3_parts[:,0])
        return self.low_syncopation

    def get_mid_syncopation(self):
        # Get syncopation of mid parts - summed snare and tom parts
        self.mid_syncopation = self.get_syncopation_1part(self.groove_3_parts[:,1])
        return self.mid_syncopation

    def get_high_syncopation(self):
        # Get syncopation of high parts - summed cymbals
        self.high_syncopation = self.get_syncopation_1part(self.groove_3_parts[:,2])
        return self.high_syncopation

    def get_density(self, part): #todo: rename to be consistent with other features? that use 1 part vs all parts functions
        # Get density of any single kit part or part group. Difference to total density is that you divide by
        # number of metrical steps, instead of total number of possible onsets in the pattern

        step_count = part.shape[0]
        onset_count = np.count_nonzero(np.ceil(part) == 1)
        average_velocity = part[part!=0.0].mean()

        if np.isnan(average_velocity):
            average_velocity = 0.0
        density = average_velocity * float(onset_count) / float(step_count)
        return density

    def get_low_density(self):
        # Get density of low part (kick)

        self.low_density = self.get_density(self.groove_3_parts[:,0])
        return self.low_density

    def get_mid_density(self):
        # Get density of mid parts (toms and snare)
        self.mid_density = self.get_density(self.groove_3_parts[:,1])
        return self.mid_density

    def get_high_density(self):
        # Get density of high parts (cymbals)
        self.high_density = self.get_density(self.groove_3_parts[:,2])
        return self.high_density

    def get_total_density(self):
        # Get total density calculated over 9 parts.
        # Total density = number of onsets / number of possible onsets (= length of pattern x 9)
        # Return values tend to be very low for this, due to high numbers of parts meaning sparse
        # matricies.

        total_step_count = self.groove_9_parts.size
        onset_count = np.count_nonzero(np.ceil(self.groove_9_parts) == 1)
        average_velocity = self.groove_9_parts[self.groove_9_parts!=0.0].mean()
        self.total_density = average_velocity * float(onset_count) / float(total_step_count)
        return self.total_density

    def get_hiness(self):
        # might need to x total density by 10 as it's not summed over one line
        self.hiness = (float(self.get_high_density()) / float(self.get_total_density()))
        return self.hiness

    def get_hisyncness(self):
        high_density = self.get_high_density()
        highParts = np.vstack([self.groove_9_parts[:,2], self.groove_9_parts[:,3], self.groove_9_parts[:,4],
                             self.groove_9_parts[:,5]])

        if high_density != 0.:
            self.hisyncness = float(self.get_high_syncopation()) / float(high_density)
        else:
            self.hisyncness = 0
        return self.hisyncness

    def get_complexity_1part(self, part):
        # Get complexity of one part. Calculated following Sioros and Guedes (2011) as combination of denisty and syncopation
        # Uses monophonic syncopation measure
        density = self.get_density(part)
        syncopation = self.get_syncopation_1part(part)
        complexity = math.sqrt(pow(density, 2) + pow(syncopation,2))
        return complexity

    def get_total_complexity(self):
        density = self.get_total_density()
        syncopation = self.get_combined_syncopation()
        self.total_complexity = math.sqrt(pow(density, 2) + pow(syncopation,2))
        return self.total_complexity

    def _get_autocorrelation_curve(self, part):
        # todo:replace this autocorrelation function for a better one
        # Return autocorrelation curve for a single part.
        # Uses autocorrelation plot function within pandas

        plt.figure()
        ax = autocorrelation_plot(part)
        autocorrelation = ax.lines[5].get_data()[1]
        plt.plot(range(1, self.groove_9_parts.shape[0]+1),
                 autocorrelation)  # plots from 1 to 32 inclusive - autocorrelation starts from 1 not 0 - 1-32
        plt.clf()
        plt.cla()
        plt.close()
        old = np.correlate(part, part, mode='full')
        return np.nan_to_num(autocorrelation)

    def get_total_autocorrelation_curve(self):
        # Get autocorrelation curve for all parts summed.

        self.total_autocorrelation_curve = 0.0
        for i in range(self.groove_9_parts.shape[1]):
            self.total_autocorrelation_curve += self._get_autocorrelation_curve(self.groove_9_parts[:,i])
        ax = autocorrelation_plot(self.total_autocorrelation_curve)

        #plt.figure()
        #plt.plot(range(1,33),self.total_autocorrelation_curve)
        return self.total_autocorrelation_curve

    def get_autocorrelation_skew(self):
        # Get skewness of autocorrelation curve
        if self.total_autocorrelation_curve:
            pass
        else:
            self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()

        self.autocorrelation_skew = stats.skew(self.total_autocorrelation_curve)
        return self.autocorrelation_skew

    def get_autocorrelation_max_amplitude(self):
        # Get maximum amplitude of autocorrelation curve
        self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()

        self.autocorrelation_max_amplitude = self.total_autocorrelation_curve.max()
        return self.autocorrelation_max_amplitude

    def get_autocorrelation_centroid(self):
        # Like spectral centroid - weighted meean of frequencies in the signal, magnitude = weights.
        centroid_sum = 0
        total_weights = 0
        self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()

        for i in range(self.total_autocorrelation_curve.shape[0]):
            # half wave rectify
            addition = self.total_autocorrelation_curve[i] * i # sum all periodicities in the signal
            if addition >= 0:
                total_weights += self.total_autocorrelation_curve[i]
                centroid_sum += addition
        if total_weights != 0:
            self.autocorrelation_centroid = centroid_sum / total_weights
        else:
            self.autocorrelation_centroid = self.groove_9_parts.shape[0] / 2
        return self.autocorrelation_centroid

    def get_autocorrelation_harmonicity(self):
        # Autocorrelation Harmonicity adapted from Lartillot et al. 2008
        self.total_autocorrelation_curve = self.get_total_autocorrelation_curve()

        alpha = 0.15
        rectified_autocorrelation = self.total_autocorrelation_curve.copy()
        for i in range(self.total_autocorrelation_curve.shape[0]):
            if self.total_autocorrelation_curve[i] < 0:
                rectified_autocorrelation[i] = 0
        
        # Fixed for Python 3: find_peaks returns a tuple (peaks, properties)
        peaks_info = find_peaks(rectified_autocorrelation)
        peaks = peaks_info[0] + 1  # peaks = lags

        inharmonic_sum = 0.0
        inharmonic_peaks = []
        for i in range(len(peaks)):
            remainder1 = 16 % peaks[i]
            if remainder1 > 16 * alpha and remainder1 < 16 * (1-alpha):
                inharmonic_sum += rectified_autocorrelation[peaks[i] - 1]  # add magnitude of inharmonic peaks
                inharmonic_peaks.append(rectified_autocorrelation[peaks[i] - 1])  # Fixed index

        harmonicity = math.exp((-0.25 * len(peaks) * inharmonic_sum / float(rectified_autocorrelation.max())))
        return harmonicity

    def _get_symmetry(self, part):
        # Calculate symmetry for any number of parts.
        # Defined as the the number of onsets that appear in the same positions in the first and second halves
        # of the pattern, divided by the total number of onsets in the pattern. As perfectly symmetrical pattern
        # would have a symmetry of 1.0

        symmetry_count = 0.0
        part1,part2 = np.split(part,2)
        for i in range(part1.shape[0]):
            for j in range(part1.shape[1]):
                if part1[i,j] != 0.0 and part2[i,j] != 0.0:
                    symmetry_count += (1.0 - abs(part1[i,j] - part2[i,j]))
        symmetry = symmetry_count*2.0 / np.count_nonzero(part)
        return symmetry

    def get_total_symmetry(self):
        # Get total symmetry of pattern. Defined as the number of onsets that appear in the same positions in the first
        # and second halves of the pattern, divided by total number of onsets in the pattern.

        self.total_symmetry = self._get_symmetry(self.groove_9_parts)
        return self.total_symmetry

class MicrotimingFeatures():
    def __init__(self, microtiming_matrix, tempo):
        self.microtiming_matrix = microtiming_matrix

        self.tempo = tempo
        self.average_timing_matrix = self.get_average_timing_deviation()
        self._get_swing_info()
        self.microtiming_event_profile = np.hstack([self._getmicrotiming_event_profile_1bar(self.microtiming_matrix[0:16]),
                                            self._getmicrotiming_event_profile_1bar(self.microtiming_matrix[16:])])
        self.laidback_events = self._get_laidback_events()
        self.pushed_events = self._get_pushed_events()
        self.is_swung = self.check_if_swung()


    def calculate_all_features(self):
        # Get all microtiming features.
        self.laidbackness = self.laidback_events - self.pushed_events
        self.timing_accuracy = self.get_timing_accuracy()

    def get_all_features(self) -> dict:
        #todo: doesn't return tripletness
        self.calculate_all_features()
        return {
            "is_swung": self.is_swung,
            "swingness": self.swingness,
            "laidbackness": self.laidbackness,
            "timing_accuracy": self.timing_accuracy,
        }
  
    def check_if_swung(self):
        # Check if loop is swung - return 'true' or 'false'

        if self.swingness > 0.0:
            self.is_swung = 1
        elif self.swingness == 0.0:
            self.is_swung = 0

        return self.is_swung

    def get_swing_ratio(self):
        #todo: implement
        pass

    def get_swingness(self):
        return self.swingness

    def get_laidbackness(self):
        self.laidbackness = self.laidback_events - self.pushed_events
        return self.laidbackness

    def _get_swing_info(self):
        # Calculate all of the swing characteristics (swing ratio, swingness etc) in one go

        swung_note_positions = list(range(self.average_timing_matrix.shape[0]))[3::4]

        swing_count = 0.0
        j = 0
        for i in swung_note_positions:
            if self.average_timing_matrix[i] < -25.0:
                swing_count +=1
            j+=1

        swing_count = np.clip(swing_count,0,len(swung_note_positions))

        if swing_count >0:
            self.swingness = (1 + (swing_count / len(swung_note_positions)/9)) #todo: weight swing count
        else:
            self.swingness = 0.0


    def _getmicrotiming_event_profile_1bar(self, microtiming_matrix):
        # Get profile of microtiming events for use in pushness/laidbackness/ontopness features
        # This profile represents the presence of specific timing events at certain positions in the pattern
        # Microtiming events fall within the following categories:
        #   Kick timing deviation - before/after metronome, before/after hihat, beats 1 and 3
        #   Snare timing deviation - before/after metronome, before/after hihat, beats 2 and 4
        # As such for one bar the profile contains 16 values.
        # The profile uses binary values - it only measures the presence of timing events, and the style features are
        # then calculated based on the number of events present that correspond to a certain timing feel.

        timing_to_grid_profile = np.zeros([8])
        timing_to_cymbal_profile = np.zeros([8])
        threshold = 12.0
        kick_timing_1 = microtiming_matrix[0, 0]
        hihat_timing_1 = microtiming_matrix[0, 2]
        snareTiming2 = microtiming_matrix[4, 1]
        hihat_timing_2 = microtiming_matrix[4, 2]
        kick_timing_3 = microtiming_matrix[8, 0]
        hihat_timing_3 = microtiming_matrix[8, 2]
        snareTiming4 = microtiming_matrix[12, 1]
        hihat_timing_4 = microtiming_matrix[12, 2]

        if kick_timing_1 > threshold :
            timing_to_grid_profile[0] = 1
        if kick_timing_1 < -threshold:
            timing_to_grid_profile[1] = 1
        if snareTiming2 > threshold:
            timing_to_grid_profile[2] = 1
        if snareTiming2 < -threshold:
            timing_to_grid_profile[3] = 1

        if kick_timing_3 > threshold:
            timing_to_grid_profile[4] = 1
        if kick_timing_3 < -threshold:
            timing_to_grid_profile[5] = 1
        if snareTiming4 > threshold:
            timing_to_grid_profile[6] = 1
        if snareTiming4 < -threshold:
            timing_to_grid_profile[7] = 1

        if kick_timing_1 > hihat_timing_1 + threshold:
            timing_to_cymbal_profile[0] = 1
        if kick_timing_1 < hihat_timing_1 - threshold:
            timing_to_cymbal_profile[1] = 1
        if snareTiming2 > hihat_timing_2 + threshold:
            timing_to_cymbal_profile[2] = 1
        if snareTiming2 < hihat_timing_2 - threshold:
            timing_to_cymbal_profile[3] = 1

        if kick_timing_3 > hihat_timing_3 + threshold:
            timing_to_cymbal_profile[4] = 1
        if kick_timing_3 < hihat_timing_3 - threshold:
            timing_to_cymbal_profile[5] = 1
        if snareTiming4 > hihat_timing_4 + threshold:
            timing_to_cymbal_profile[6] = 1
        if snareTiming4 < hihat_timing_4 - threshold:
            timing_to_cymbal_profile[7] = 1

        microtiming_event_profile_1bar = np.clip(timing_to_grid_profile+timing_to_cymbal_profile,0,1)

        return microtiming_event_profile_1bar

    def _get_pushed_events(self):
        # Calculate how 'pushed' the loop is, based on number of pushed events / number of possible pushed events

        push_events = self.microtiming_event_profile[1::2]
        push_event_count = np.count_nonzero(push_events)
        total_push_positions = push_events.shape[0]
        self.pushed_events = push_event_count / total_push_positions
        return self.pushed_events

    def _get_laidback_events(self):
        # Calculate how 'laid-back' the loop is, based on the number of laid back events / number of possible laid back events

        laidback_events = self.microtiming_event_profile[0::2]
        laidback_event_count = np.count_nonzero(laidback_events)
        total_laidback_positions = laidback_events.shape[0]
        self.laidback_events =  laidback_event_count / float(total_laidback_positions)
        return self.laidback_events

    def get_timing_accuracy(self):
        # Calculate timing accuracy of the loop

        swung_note_positions = list(range(self.average_timing_matrix.shape[0]))[3::4]
        nonswing_timing = 0.0
        nonswing_note_count = 0
        triplet_positions = 1, 5, 9, 13, 17, 21, 25, 29

        for i in range(self.average_timing_matrix.shape[0]):
            if i not in swung_note_positions and i not in triplet_positions:
                if ~np.isnan(self.average_timing_matrix[i]):
                    nonswing_timing += abs(np.nan_to_num(self.average_timing_matrix[i]))
                    nonswing_note_count += 1
        self.timing_accuracy = nonswing_timing / float(nonswing_note_count)

        return self.timing_accuracy

    def get_average_timing_deviation(self):
        # Get vector of average microtiming deviation at each metrical position

        self.average_timing_matrix = np.zeros([self.microtiming_matrix.shape[0]])
        for i in range(self.microtiming_matrix.shape[0]):
            row_sum = 0.0
            hit_count = 0.0
            rowIsEmpty = np.all(np.isnan(self.microtiming_matrix[i,:]))
            if rowIsEmpty:
                self.average_timing_matrix[i] = np.nan
            else:
                for j in range(self.microtiming_matrix.shape[1]):
                    if np.isnan(self.microtiming_matrix[i,j]):
                        pass
                    else:
                        row_sum += self.microtiming_matrix[i,j]
                        hit_count += 1.0
                self.average_timing_matrix[i] = row_sum / hit_count
        return self.average_timing_matrix


if __name__ == "__main__":
    import pickle
    import os
    import numpy as np
    from data.feature import DrumMIDIFeature
    import sounddevice as sd

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(project_root, "dataset", "serialized", "50sAutumn.pkl")
    with open(test_path, "rb") as f:
        pickle_data = pickle.load(f)

    rand_idx = np.random.randint(0, len(pickle_data))
    sample = pickle_data[rand_idx]
    feature = DrumMIDIFeature(sample["midi_bytes"])
    descriptors = FeatureDescriptors(feature)
    descriptors.print_all_features()
    feature.play()






