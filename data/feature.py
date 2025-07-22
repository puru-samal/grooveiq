import os
from symusic import Score, Note, core
from .utils import render, get_num_bars, get_num_quarters
_HAS_SOUNDDEVICE = False
try:
    import sounddevice as sd
    _HAS_SOUNDDEVICE = True
except Exception as e:
    print(f"Error importing sounddevice: {e}")
    print("Will not be able to play audio")
import numpy as np
import matplotlib.pyplot as plt
import torch
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import random
import pretty_midi

"""
Basic usage:

# Load a MIDI file
feature = DrumMIDIFeature.from_file(midi_path, drum_map=drum_map)

# Play the MIDI file
feature.play()

# Convert the MIDI file to a fixed grid representation
fixed_grid = feature.to_fixed_grid(steps_per_quarter=4)

# Convert the fixed grid representation back to a MIDI file
feature_from_grid = feature.from_fixed_grid(fixed_grid, steps_per_quarter=4)

# Play the MIDI file
feature_from_grid.play()

"""

class DrumMIDIFeature:
    """
    Feature extraction and audio rendering for a single Drum MIDI file.

    Attributes:
        score (Score): The parsed MIDI score.
        sf_path (str): Path to the default SoundFont for synthesis.
    """
    # Path to the soundfont file
    sf_path: str = os.path.join(os.path.dirname(os.path.abspath(__file__)), "soundfonts", "Standard_Drum_Kit.sf2")
    canonical_map: dict = {
        36: 'kick',
        38: 'snare',
        42: 'hh_closed',
        46: 'hh_open',
        43: 'low_tom',
        47: 'mid_tom',
        50: 'high_tom',
        49: 'crash',
        51: 'ride',
    }
    
    def __init__(self, midi_bytes: bytes, drum_map: dict = None) -> None:
        """
        Initialize the feature extractor for a given MIDI file.

        Expects midi pitches to belong to the canonical map. If they don't then you must provide a drum_map
        mapping the midi pitches to the canonical map.
        
        WARNING: if you dont provide a drum_map and there are notes in the MIDI file that are not in the canonical map,
        they will be dropped.

        Args:
            midi_bytes (bytes): Raw MIDI data as bytes.
            drum_map (dict): A dictionary mapping canonical MIDI pitches to list of MIDI notes.
            
            canonical MIDI pitch: the pitch that is used to represent the drum class.
            - 36: kick
            - 38: snare
            - 42: hh_closed
            - 46: hh_open
            - 43: low_tom
            - 47: mid_tom
            - 50: high_tom
            - 49: crash
            - 51: ride
            
            Mapping format:
            {
                canonical_MIDI_pitch: {
                    'midi_notes': [MIDI_notes]
                }
            }
        Raises:
            ValueError: If the MIDI file does not meet Groove MIDI assumptions.
        """
        self.score = Score.from_midi(midi_bytes)
        self.drum_map = drum_map
        # Map from MIDI notes to canonical MIDI pitches
        self.inv_drum_map = {}
        if not self.is_valid():
            print(f"Error: Invalid file!")
            print(f"Time signatures: {self.score.time_signatures}")
            print(f"Tempos: {self.score.tempos}")
            raise ValueError("Failed checks:", self.validity_report())
            
        # Optional: Simplify the drum representation
        self.num_dropped_notes = 0  # Number of notes dropped due to invalid MIDI notes
        self.dropped_set = set()    # Set of MIDI notes that are dropped

        if self.drum_map is not None:
            # Create inverse drum map
            # This is used to map midi notes to canonical MIDI classes
            for target_note, map_info in self.drum_map.items():
                for source_note in map_info['midi_notes']:
                    self.inv_drum_map[source_note] = target_note

            # Simplify drum notes
            for note in self.score.tracks[0].notes:
                if note.pitch in self.inv_drum_map:
                    # Calculate the pitch offset needed
                    offset = self.inv_drum_map[note.pitch] - note.pitch
                    note.shift_pitch(offset, inplace=True)
                else:
                    self.num_dropped_notes += 1
                    self.dropped_set.add(note.pitch)

        self.score.tracks[0].notes.filter(lambda x: x.pitch in self.canonical_map.keys(), inplace=True)
        self.score.tracks[0].notes.sort(key=None, reverse=False, inplace=True)
        # End time in ticks of the feature
        # Is the start time of the last note event
        self.end = self.score.tracks[0].notes[-1].time

    #########################################################################################
    # Validity checks
    #########################################################################################

    def is_valid(self) -> bool:
        """
        Returns True if all validity checks pass, False otherwise.
        """
        return len(self.validity_report()) == 0

    def validity_report(self) -> list:
        """
        Returns a list of failed validity checks for the MIDI file.
        If the list is empty, the file is valid.
        """
        checks = {
            "starts_at_zero": self.score.start() == 0,
            "has_single_track": len(self.score.tracks) == 1,
            "track_is_drum": len(self.score.tracks) == 1 and self.score.tracks[0].is_drum,
            "has_single_tempo": len(self.score.tempos) == 1 and self.score.tempos[0].time == 0,
            "has_constant_time_signature": len(self.score.time_signatures) == 1 or all(ts.numerator == self.score.time_signatures[0].numerator and ts.denominator == self.score.time_signatures[0].denominator for ts in self.score.time_signatures),
            "drum_map_or_notes_are_in_canonical_map": self.drum_map is not None or all(note.pitch in self.canonical_map.keys() for note in self.score.tracks[0].notes)
        }
        return [name for name, passed in checks.items() if not passed]

    #########################################################################################
    # I/O methods
    #########################################################################################

    @classmethod
    def from_file(cls, file_path: str, drum_map: dict = None) -> "DrumMIDIFeature":
        """
        Create a DrumMIDIFeature from a MIDI file path.
        
        Args:
            file_path (str): Path to the MIDI file.
            drum_map (dict): A dictionary mapping MIDI pitches to (reduced) drum indices.
            
        Returns:
            DrumMIDIFeature: A new instance created from the file.
            
        Raises:
            FileNotFoundError: If the file doesn't exist.
            ValueError: If the MIDI file is invalid.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"MIDI file not found: {file_path}")
        
        with open(file_path, 'rb') as f:
            midi_bytes = f.read()
        
        return cls(midi_bytes, drum_map)

    @classmethod
    def from_score(cls, score: Score, drum_map: dict = None) -> "DrumMIDIFeature":
        """
        Create a DrumMIDIFeature from a symusic Score object.
        
        Args:
            score (Score): A symusic Score object.
            reduced_drum_map (bool): Whether to use the reduced drum map.
            
        Returns:
            DrumMIDIFeature: A new instance created from the score.
        """
        midi_bytes = score.dumps_midi()
        return cls(midi_bytes, drum_map)

    @classmethod
    def from_base64(cls, base64_string: str, drum_map: dict = None) -> "DrumMIDIFeature":
        """
        Create a DrumMIDIFeature from a base64 encoded MIDI string.
        
        Args:
            base64_string (str): Base64 encoded MIDI data.
            drum_map (dict): A dictionary mapping MIDI pitches to (reduced) drum indices.
            
        Returns:
            DrumMIDIFeature: A new instance created from the base64 string.
        """
        import base64
        midi_bytes = base64.b64decode(base64_string)
        return cls(midi_bytes, drum_map)

    def to_pretty_midi(self) -> pretty_midi.PrettyMIDI:
        """
        Convert the DrumMIDIFeature to a pretty_midi.PrettyMIDI object.
        """
        pm_obj = pretty_midi.PrettyMIDI(initial_tempo=self.score.tempos[0].tempo)
        pm_obj.time_signature_changes.append(pretty_midi.TimeSignature(numerator=self.score.time_signatures[0].numerator, denominator=self.score.time_signatures[0].denominator, time=0))
        score_in_seconds = self.score.to('second')
        pm_obj.instruments.append(pretty_midi.Instrument(program=0, is_drum=True, name="Drums"))
        for note in score_in_seconds.tracks[0].notes:
            pm_obj.instruments[0].notes.append(pretty_midi.Note(
                velocity=note.velocity,
                pitch=note.pitch,
                start=note.time,
                end=note.time + note.duration,
            ))
        return pm_obj

    #########################################################################################
    # Segmentation methods
    #########################################################################################

    def split_segments(self, time_signature: tuple[int, int], num_bars: int) -> Tuple[list["DrumMIDIFeature"], int]:
        """
        Split the score into segments using a sliding window of num_bars.

        Args:
            time_signature: Tuple (beats per bar, beat resolution denominator).
            num_bars: Number of bars in each segment.

        Returns:
            List of DrumMIDIFeature segments.
            Number of errors.
        """
        beats_per_bar = time_signature[0]
        beat_resolution = time_signature[1]
        ticks_per_beat = self.score.tpq * (4 / beat_resolution)
        ticks_per_bar = int(beats_per_bar * ticks_per_beat)

        if self.end == 0:  # Edge case: only one note at start
            total_bars = 1
        else:
            total_bars = get_num_bars(self.end, time_signature, self.score.tpq)

        total_ticks = total_bars * ticks_per_bar
        segment_ticks = num_bars * ticks_per_bar

        segments = []
        num_errors = 0

        # Start positions: 0, ticks_per_bar, ..., up to last valid start
        for start_tick in range(0, total_ticks - segment_ticks + 1, ticks_per_bar):
            end_tick = start_tick + segment_ticks

            score_cpy = self.score.copy()
            score_cpy.clip(start_tick, end_tick, clip_end=True, inplace=True)
            score_cpy.shift_time(-start_tick, inplace=True)

            try:
                feature = DrumMIDIFeature.from_score(score_cpy)
                segments.append(feature)
            except Exception as e:
                print(f"Error creating feature: {e}")
                num_errors += 1
                continue

        return segments, num_errors


    def get_random_segment(self, time_signature: tuple[int, int], num_bars: int) -> "DrumMIDIFeature":
        """
        Get a random segment of the MIDI file.
        Args:
            time_signature: Time signature of the segment.
            num_bars: Number of bars in the segment.
        Returns:
            DrumMIDIFeature: A new instance created from the segment.
        """
        beats_per_bar = time_signature[0]
        beat_resolution = time_signature[1]
        ticks_per_beat = self.score.tpq * (4 / beat_resolution)
        ticks_per_bar = int(beats_per_bar * ticks_per_beat)
        if self.end == 0: # Edge case: MIDI file contains only one note at start time 0 (eg. only one crash at start time 0)
            total_bars = 1
        else:
            total_bars = get_num_bars(self.end, time_signature, self.score.tpq)
        
        # TODO: handle this case (concatenate multiple duplicate segments)
        if total_bars <= num_bars:
            # Concatenate multiple duplicate segments
            num_duplicates = num_bars // total_bars
            start_time = 0
            end_time = total_bars * ticks_per_bar
            score_cpy = self.score.copy()
            score_cpy.clip(start_time, end_time, clip_end=True, inplace=True)
            for i in range(1, num_duplicates):
                seg_cpy = score_cpy.copy()
                seg_cpy.shift_time(i * end_time, inplace=True)
                for note in seg_cpy.tracks[0].notes:
                    score_cpy.tracks[0].notes.append(note)
            return DrumMIDIFeature.from_score(score_cpy)
        
        start_time = np.random.randint(0, total_bars - num_bars) * ticks_per_bar
        end_time = start_time + num_bars * ticks_per_bar
        score_cpy = self.score.copy()
        score_cpy.clip(start_time, end_time, clip_end=True, inplace=True)
        score_cpy.shift_time(-start_time, inplace=True)
        return DrumMIDIFeature.from_score(score_cpy)
    
    def _get_group(self, pitch: int) -> int:
        """
        Get the group of a pitch.
        """
        low_pitches = [36, 43] # kick, low_tom
        medium_pitches = [38, 47, 50] # snare, mid_tom, high_tom
        high_pitches = [42, 46, 49, 51] # hh_closed, hh_open, crash, ride
        if pitch in low_pitches:
            return 0
        elif pitch in medium_pitches:
            return 1
        elif pitch in high_pitches:
            return 2
        else:
            raise ValueError(f"Pitch {pitch} not found in any group.")

    #########################################################################################
    # Fixed grid methods
    #########################################################################################

    def to_fixed_grid(self, steps_per_quarter: int = 4) -> Tuple[torch.Tensor, Dict]:
        """
        Convert a full MIDI performance into a fixed grid of shape (T, E, M),
        where T = total number of quantized steps based on steps_per_quarter.
        
        Args:
            steps_per_quarter: Number of steps per quarter note (e.g., 4 = 16th note resolution)
        
        Returns:
            - grid: (T, E, M) tensor with hit/velocity/offset
                    where,
                    T = total number of quantized steps based on steps_per_quarter.
                    E = number of drum classes
                    M = [hit, velocity, offset]
                    - hit: 1 if the drum is hit, 0 otherwise
                    - velocity: velocity of the drum
                    - offset: offset of the drum
            - stats: dict with note loss statistics:  
        """
        time_signature = (self.score.time_signatures[0].numerator, self.score.time_signatures[0].denominator)
        T = get_num_quarters(self.end, time_signature, self.score.tpq) * steps_per_quarter + 1
        note_to_index = {note: i for i, note in enumerate(self.canonical_map.keys())}
        E = len(note_to_index)
        M = 3
        grid = torch.zeros((T, E, M))
        stats = {
            'total_notes': 0,
            'note_loss': 0
        }
        
        for note in self.score.tracks[0].notes:
            if note.pitch not in note_to_index:
                raise ValueError(f"MIDI note {note.pitch} not found in drum map. Please check the drum map.")
            
            stats['total_notes'] += 1
            index = note_to_index[note.pitch]
            t_exact = (note.time / self.score.tpq) * steps_per_quarter
            t = int(np.round(t_exact))
            offset = t_exact - t
            velocity = note.velocity / 127.0

            # Largest velocity wins
            if grid[t, index, 0] == 1.0:
                stats['note_loss'] += 1
                # Largest velocity wins
                if velocity <= grid[t, index, 1]:
                    continue

            # Update cell with this (stronger) note
            grid[t, index] = torch.tensor([1.0, velocity, offset])

        reorder_idx = [0, 4, 1, 5, 6, 2, 3, 7, 8] # [[low], [mid], [high]]
        grid = grid[:, reorder_idx, :]
        
        return grid.float(), stats
    
    
    def from_fixed_grid(self, grid: torch.Tensor, steps_per_quarter: int) -> "DrumMIDIFeature":
        """
        Convert a fixed grid tensor back to a DrumMIDIFeature object.
        Args:
            grid: Tensor of shape (T, E, M)
            steps_per_quarter: Number of steps per quarter note
            tpq: Ticks per quarter note for the MIDI
            tempo: Tempo in BPM
        Returns:
            DrumMIDIFeature instance
        """
        reorder_idx = [0, 2, 5, 6, 1, 3, 4, 7, 8]
        grid = grid[:, reorder_idx, :]
        T, E, M = grid.shape
        score = self.score.copy()
        track = score.tracks[0]
        track.notes = []
        tpq = score.tpq

        # Build reverse drum map: index -> pitch
        index_to_note = {i: note for i, note in enumerate(self.canonical_map.keys())}

        for t in range(T):
            for e in range(E):
                hit, velocity, offset = grid[t, e].tolist()
                if hit > 0:
                    pitch = index_to_note[e]
                    # Calculate time in ticks
                    t_exact = t + offset
                    time = int(round((t_exact / steps_per_quarter) * tpq))
                    if time < 0:
                        time = 0
                    note = Note(
                        pitch=pitch,
                        time=time,
                        duration=tpq // steps_per_quarter,  # 1 grid step duration
                        velocity=int(velocity * 127)
                    )
                    track.notes.append(note)


        score.tracks[0] = track
        return DrumMIDIFeature.from_score(score)
    
    def simplify_fixed_grid(self, win_sizes: List[tuple] = [(1, 0.1), (2, 0.5), (3, 0.15), (4, 0.25)], velocity_range: tuple[float, float] = (0.5, 0.8), max_hits_per_win: int = 1, win_retain_prob: float = 0.8) -> "DrumMIDIFeature":
        """
        Simplifies a (T, E, 3) HVO grid by selecting up to `max_hits_per_window` strongest hits per window,
        applying thresholding and random retention.
        
        Args:
            fixed_grid (Tensor): (T, E, 3) drum sequence (hit, velocity, offset)
            win_sizes (List[tuple(int, float)]): List of window sizes where each tuple is (window_size, prob) (e.g. [(2, 0.5), (4, 0.5)])
            velocity_range (tuple(float, float)): range to sample velocity threshold from
            max_hits_per_window (int): max hits to keep per window (randomized)
            win_retain_prob (float): probability to keep any given window (to allow full zeroing)

        Returns:
            Tensor: (T, E, 3) simplified grid
        """
        fixed_grid, _ = self.to_fixed_grid(steps_per_quarter=4)
        T= fixed_grid.shape[0]
        out_grid = torch.zeros_like(fixed_grid)
        win_size = random.choices([win_size for win_size, _ in win_sizes], weights=[prob for _, prob in win_sizes], k=1)[0]

        # Compute the max velocity across the entire performance
        max_velocity = fixed_grid[:, :, 1].max().item()
        velocity_threshold_ratio = random.uniform(velocity_range[0], velocity_range[1])
        velocity_threshold = max_velocity * velocity_threshold_ratio

        for i in range(0, T, win_size):
            slice = fixed_grid[i:i+win_size]  # shape: (win_size, E, 3)

            # Get velocities and apply threshold
            velocity_slice = slice[:, :, 1]  # shape: (win_size, E)
            mask = velocity_slice > velocity_threshold

            if not mask.any():
                continue  # skip empty window

            # Flatten valid indices
            valid_indices = mask.nonzero(as_tuple=False)  # shape: (N_valid, 2), with (t, e)

            # Randomly choose to keep this window
            if random.random() > win_retain_prob:
                continue  # drop entire window randomly

            # Determine how many hits to keep (could be 1 or up to max_hits_per_window)
            k = min(len(valid_indices), random.randint(1, max_hits_per_win))

            # Rank valid hits by velocity
            velocities = velocity_slice[mask]  # 1D tensor
            topk = torch.topk(velocities, k)
            selected_indices = valid_indices[topk.indices]  # shape: (k, 2)

            for t_rel, e in selected_indices:
                out_grid[i + t_rel, e] = slice[t_rel, e]  # copy full HVO

        return self.from_fixed_grid(out_grid, steps_per_quarter=4)
    
    def reduce_groove_fixed_grid(self, velocity_threshold=0.4):
        """
        Reduces a fixed-grid HVO representation by removing ornamental hits based on metrical salience.
        Assumes 1/16th resolution for 4/4 time.

        Args:
            grid (Tensor): (T, E, 3) input grid [hit, velocity, offset]
            velocity_threshold (float): Minimum velocity to retain a note

        Returns:
            Tensor: Reduced grid (T, E, 3), with ornamental hits removed or shifted
        """
        grid, _ = self.to_fixed_grid(steps_per_quarter=4)

        pattern = torch.tensor([0, -2, -1, -2], device=grid.device)
        repeats = (grid.shape[0] + len(pattern) - 1) // len(pattern)  # ceil division

        metrical_profile = pattern.repeat(repeats)[:grid.shape[0]]  # Trim to T
        reduced_grid = grid.clone()

        T, E, _ = grid.shape

        for e in range(E):
            part = reduced_grid[:, e, :]  # shape: (T, 3)
            velocity = part[:, 1]

            # Step 1: Remove ghost notes
            part[velocity <= velocity_threshold] = 0.0

            for i in range(T):
                if part[i, 0] != 0:  # there's a hit
                    for k in range(max(0, i - 3), i):
                        if part[k, 0] != 0 and metrical_profile[k] < metrical_profile[i]:
                            # Find strongest prior pulse before k
                            prev = 0
                            for l in range(k):
                                if part[l, 0] != 0:
                                    prev = l

                            m_strength = metrical_profile[prev:k].max()
                            if m_strength <= metrical_profile[k]:
                                # Density transform: remove k
                                part[k] = 0.0
                            else:
                                # Figural shift: move k to stronger m
                                m_idx = prev + torch.argmax(metrical_profile[prev:k])
                                part[m_idx] = part[k]
                                part[k] = 0.0

                if part[i, 0] == 0:
                    for k in range(max(0, i - 3), i):
                        if part[k, 0] != 0 and metrical_profile[k] < metrical_profile[i]:
                            # Syncopation correction
                            part[i] = part[k]
                            part[k] = 0.0

            reduced_grid[:, e, :] = part

        return reduced_grid
    
    def to_button_hvo(self, steps_per_quarter: int = 4, num_buttons: int = 3) -> Tuple[torch.Tensor, Dict]:
        """
        Convert a fixed grid to a button HVO grid.
        """
        grid, _ = self.to_fixed_grid(steps_per_quarter=steps_per_quarter)
        T, E, M = grid.shape
        button_hvo = torch.zeros((T, num_buttons, 3))
        button_map = {
            0: [0, 1],       # low
            1: [2, 3, 4],    # mid
            2: [5, 6, 7, 8], # high
        }

        inv_button_map = {v: k for k, V in button_map.items() for v in V}

        # Get the number of hits per button
        for t in range(T):
            for e in range(E):
                hit, velocity, offset = grid[t, e].tolist()
                if hit > 0:
                    button_e = inv_button_map[e]
                    if button_e < num_buttons and button_hvo[t, button_e, 1] < velocity:
                        button_hvo[t, button_e] = torch.tensor([1.0, velocity, offset])
        return button_hvo.float()
    
    def simplify_to_button_hvo(self, steps_per_quarter=4, num_buttons=3,
                           win_sizes=[(1, 0.1), (2, 0.5), (3, 0.15), (4, 0.25)],
                           velocity_range=(0.5, 0.8), max_hits_per_win=1, win_retain_prob=0.8) -> torch.Tensor:
        """
        Combines simplification and button HVO projection into a single pass.
        Returns:
            button_hvo: (T, num_buttons, 3)
        """
        grid, _ = self.to_fixed_grid(steps_per_quarter=steps_per_quarter)
        T, E, M = grid.shape

        # Mapping for buttons
        button_map = {
            0: [0, 1],       # low
            1: [2, 3, 4],    # mid
            2: [5, 6, 7, 8], # high
        }
        inv_button_map = {e: k for k, v in button_map.items() for e in v}
        button_hvo = torch.zeros((T, num_buttons, 3), device=grid.device)

        # Sample simplification parameters
        win_size = random.choices([s for s, _ in win_sizes], weights=[p for _, p in win_sizes])[0]
        max_velocity = grid[:, :, 1].max().item()
        velocity_threshold = max_velocity * random.uniform(*velocity_range)

        for i in range(0, T, win_size):
            slice = grid[i:i+win_size]  # (w, E, 3)
            vel_slice = slice[:, :, 1]
            mask = vel_slice > velocity_threshold
            if not mask.any() or random.random() > win_retain_prob:
                continue

            valid_indices = mask.nonzero(as_tuple=False)  # (N_valid, 2)
            k = min(len(valid_indices), random.randint(1, max_hits_per_win))
            velocities = vel_slice[mask]
            topk = torch.topk(velocities, k)
            selected = valid_indices[topk.indices]

            for t_rel, e in selected:
                t = i + t_rel
                b = inv_button_map[int(e)]
                if b < num_buttons:
                    if button_hvo[t, b, 1] < slice[t_rel, e, 1]:  # keep stronger hit
                        button_hvo[t, b] = slice[t_rel, e]  # full HVO

        return button_hvo

    
    #########################################################################################
    # Token sequence methods
    #########################################################################################
    
    def to_token_sequence(self, steps_per_quarter: int = 4) -> Tuple[torch.Tensor, Dict]:
        """
        Convert a full MIDI performance into a token sequence.
        Each note is represented by a token of the form (pitch, velocity, grid_step, offset)
        Args:
            steps_per_quarter: Number of steps per quarter note
        Returns:
            token_sequence: Tensor of shape (T, 5)
        """
        note_to_index = {note: i for i, note in enumerate(self.canonical_map.keys())}
        token_sequence = []
        for track in self.score.tracks:
            for note in track.notes:
                if note.pitch not in note_to_index:
                    continue
                index = note_to_index[note.pitch] # [0..num_drum_classes - 1]
                velocity = note.velocity / 127.0 # [0..1]
                t_exact = (note.time / self.score.tpq) * steps_per_quarter # [0..32]
                t = int(np.round(t_exact)) # [0..33]
                offset = t_exact - t # [-0.5..0.5]
                group = self._get_group(note.pitch) # [0..2]
                token = torch.tensor([index, velocity, t, offset, group])
                token_sequence.append(token)
        token_sequence.sort(key=lambda x: (x[2], x[3])) # Sort by grid_step and offset
        token_sequence = torch.stack(token_sequence, dim=0)
        return token_sequence
    
    def from_token_sequence(self, token_sequence: torch.Tensor, steps_per_quarter: int = 4) -> "DrumMIDIFeature":
        """
        Convert a token sequence back to a DrumMIDIFeature object.
        Args:
            token_sequence: Tensor of shape (T, 5)
            steps_per_quarter: Number of steps per quarter note
        Returns:
            DrumMIDIFeature instance
        """
        score = self.score.copy()
        track = score.tracks[0]
        track.notes.clear()
        tpq = score.tpq

        # Build reverse drum map: index -> pitch
        index_to_note = {i: note for i, note in enumerate(self.canonical_map.keys())}

        for t in range(token_sequence.shape[0]):
            index, velocity, t, offset, group = token_sequence[t].tolist()
            pitch = index_to_note[index]
            time = int(round(((t + offset) / steps_per_quarter) * tpq))
            note = Note(pitch=pitch, time=time, duration=tpq // steps_per_quarter, velocity=int(velocity * 127))
            track.notes.append(note)
        
        return DrumMIDIFeature.from_score(score)
    
    #########################################################################################
    # Flexible grid methods
    #########################################################################################

    def to_flexible_grid(self, max_hits_per_class: Dict[int, int], steps_per_quarter: int = 4) -> Tuple[torch.Tensor, Dict]:
        """
        Creates a flexible grid representation using an externally provided max_hits_per_class mapping.

        Args:
            max_hits_per_class: Dict mapping drum class (pitch) to number of channels
            steps_per_quarter: Temporal resolution

        Returns:
            flexible_grid: Tensor of shape (T, E', M)
            stats: Dict with note loss statistics
        """
        time_signature = (self.score.time_signatures[0].numerator, self.score.time_signatures[0].denominator)
        T = get_num_quarters(self.end, time_signature, self.score.tpq) * steps_per_quarter + 1
        E = len(self.canonical_map)
        stats = {
            'total_notes': 0,
            'note_loss': 0
        }

        # Map from (timestep, drum class pitch) to list of (velocity, offset)
        hits_by_te = defaultdict(list)

        for note in self.score.tracks[0].notes:
            if note.pitch not in self.canonical_map:
                continue
            stats['total_notes'] += 1
            pitch = note.pitch
            t_exact = (note.time / self.score.tpq) * steps_per_quarter
            t = int(np.round(t_exact))
            offset = t_exact - t
            velocity = note.velocity / 127.0
            hits_by_te[(t, pitch)].append((velocity, offset))

        E_prime = sum(max_hits_per_class.values())
        drum_class_to_slots = {}
        current_index = 0
        for pitch in self.canonical_map.keys():
            drum_class_to_slots[pitch] = list(range(current_index, current_index + max_hits_per_class[pitch]))
            current_index += max_hits_per_class[pitch]

        grid = torch.zeros((T, E_prime, 3))

        # Fill in the grid
        for (t, pitch), hits in hits_by_te.items():
            slots = drum_class_to_slots[pitch]
            for i, (vel, off) in enumerate(hits):
                if i >= len(slots):
                    stats['note_loss'] += 1
                    continue
                grid[t, slots[i]] = torch.tensor([1.0, vel, off])

        return grid.float(), stats

    def from_flexible_grid(self, grid: torch.Tensor, max_hits_per_class: Dict[int, int], steps_per_quarter: int) -> 'DrumMIDIFeature':
        """
        Converts a flexible grid back into a DrumMIDIFeature.

        Args:
            grid: Tensor of shape (T, E', M)
            drum_class_to_slots: Mapping of drum class to flexible slot indices
            steps_per_quarter: Temporal resolution

        Returns:
            Reconstructed DrumMIDIFeature
        """
        T, E_prime, M = grid.shape
        score = self.score.copy()
        track = score.tracks[0]
        track.notes.clear()
        tpq = score.tpq

        # Reverse slot map for labeling
        slot_to_drum = {}
        current_index = 0
        for pitch in self.canonical_map.keys():
            slot_to_drum.update({s: pitch for s in range(current_index, current_index + max_hits_per_class[pitch])})
            current_index += max_hits_per_class[pitch]

        for t in range(T):
            for s in range(E_prime):
                hit, vel, offset = grid[t, s].tolist()
                if hit > 0 and s in slot_to_drum:
                    pitch = slot_to_drum[s]
                    t_exact = t + offset
                    time = int(round((t_exact / steps_per_quarter) * tpq))
                    note = Note(pitch=pitch, time=time, duration=tpq // steps_per_quarter, velocity=int(vel * 127))
                    track.notes.append(note)

        return DrumMIDIFeature.from_score(score)
    

    #########################################################################################
    # Button HVO methods
    #########################################################################################

    def from_button_hvo(self, button_hvo: torch.Tensor, steps_per_quarter: int) -> 'DrumMIDIFeature':
        """
        Convert a button HVO tensor into a DrumMIDIFeature.
        Button hits get mapped to notes in the canonical drum map.
        """
        T, E, M = button_hvo.shape
        score = self.score.copy()
        track = score.tracks[0]
        track.notes = []
        tpq = score.tpq
        
        # Build reverse drum map: index -> pitch
        index_to_note = {i: note for i, note in enumerate(self.canonical_map.keys())}

        for t in range(T):
            for e in range(E):
                hit, velocity, offset = button_hvo[t, e].tolist()
                if hit > 0:
                    pitch = index_to_note[e]
                    # Calculate time in ticks
                    t_exact = t + offset
                    time = int(round((t_exact / steps_per_quarter) * tpq))
                    if time < 0:
                        time = 0
                    note = Note(
                        pitch=pitch,
                        time=time,
                        duration=tpq // steps_per_quarter,  # 1 grid step duration
                        velocity=int(velocity * 127)
                    )
                    track.notes.append(note)

        score.tracks[0] = track
        return DrumMIDIFeature.from_score(score)
    

    def play_button_hvo(self, button_hvo: "DrumMIDIFeature", fs: int = 44100) -> None:
        """
        Play the button HVO.
        Args:
            button_hvo: DrumMIDIFeature object
            fs: Sample rate for synthesis.
        Returns:
            audio_data: Audio data (np.ndarray of shape (samples, 2))
        """
        
        if not _HAS_SOUNDDEVICE:
            print("sounddevice not found, will not be able to play audio")
            return
        
        shift_map = {pitch: 61 + 2 * i for i, pitch in enumerate(self.canonical_map.keys())}
        shifted_score = button_hvo.score.copy()
        for note in shifted_score.tracks[0].notes:
            offset = shift_map[note.pitch] - note.pitch
            note.shift_pitch(offset, inplace=True)

        shifted_score.tracks[0].notes.sort(key=None, reverse=False, inplace=True)
        audio_data = render(shifted_score, self.sf_path, fs)
        sd.play(audio_data, fs)
        sd.wait()
        return audio_data

    
    #########################################################################################
    # Play and plot methods
    #########################################################################################

    def play(self, fs: int = 44100) -> None:
        """
        Synthesize and play the score using the default SoundFont.

        Args:
            fs: Sample rate for synthesis.
        Returns:
            audio_data: Audio data (np.ndarray of shape (samples, 2))
        """
        if not _HAS_SOUNDDEVICE:
            print("sounddevice not found, will not be able to play audio")
            return
        audio_data = render(self.score, self.sf_path, fs)
        sd.play(audio_data, fs)
        sd.wait()
        return audio_data

    def fixed_grid_plot(self, ax: Optional[plt.Axes] = None, steps_per_quarter: int = 4) -> Tuple[plt.Figure, plt.Axes]:
        """
        Plot the fixed grid.
        """
        grid, _ = self.to_fixed_grid(steps_per_quarter)
        drum_names = [self.canonical_map[k] for k in self.canonical_map.keys()]
        return DrumMIDIFeature._grid_plot(grid, ax, drum_names, "Fixed Grid", "Time Step", "Drum Class")

    
    def flexible_grid_plot(self, max_hits_per_class: Dict[int, int], ax: Optional[plt.Axes] = None, steps_per_quarter: int = 4):
        """
        Plot the flexible grid.
        """
        grid, _ = self.to_flexible_grid(max_hits_per_class, steps_per_quarter)
        E = grid.shape[1]
        # Reverse slot map for labeling
        slot_to_class_slot = {}
        current_index = 0
        for pitch in self.canonical_map.keys():
            slot_to_class_slot.update({s: (pitch, i) for i, s in enumerate(range(current_index, current_index + max_hits_per_class[pitch]))})
            current_index += max_hits_per_class[pitch]

        # Create y-axis labels: "Snare (0)", "Snare (1)", etc.
        slot_labels = []
        for s in range(E):
            if s in slot_to_class_slot:
                pitch, idx = slot_to_class_slot[s]
                name = self.canonical_map[pitch]
                slot_labels.append(f"{name} ({idx})")
            else:
                slot_labels.append(f"Unknown ({s})")
        
        return DrumMIDIFeature._grid_plot(grid, ax, slot_labels, "Flexible Grid", "Time Step", "Drum Class")

    @staticmethod
    def _grid_plot(grid: torch.Tensor, ax: Optional[plt.Axes] = None, class_names: Optional[List[str]] = None, title: str = "Grid", xlabel: str = "Time Step", ylabel: str = "Drum Class"):
        """
        Unified fixed grid visualization:
        - Velocity shown as color
        - Offset shown as marker
        - Special marker if hit == 0 but velocity/offset nonzero
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import to_rgb

        modern_colors = ['#45B7D1', '#FF6B6B']  # low → high velocity
        c0 = to_rgb(modern_colors[0])
        c1 = to_rgb(modern_colors[1])

        def interpolate_color(v):
            return tuple(c0[i] * (1 - v) + c1[i] * v for i in range(3))

        T, E, M = grid.shape

        if class_names is None:
            class_names = [f"hit{i+1}" for i in range(E)]

        hits = grid[:, :, 0].numpy().T
        velocities = grid[:, :, 1].numpy().T
        offsets = grid[:, :, 2].numpy().T

        if ax is None:
            fig, ax = plt.subplots(figsize=(min(0.7 * T, 18), 6))
        else:
            fig = ax.get_figure()

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        ax.set_xlim(0, T)
        ax.set_ylim(-0.5, E - 0.5)
        ax.set_yticks(range(E))
        ax.set_yticklabels(class_names)
        ax.grid(True, axis='x', linestyle=':', alpha=0.3)

        for e in range(E):
            for t in range(T):
                hit = hits[e, t]
                velocity = velocities[e, t]
                offset = offsets[e, t]
                
                '''
                # Check for ghost articulation: no hit, but velocity or offset nonzero
                is_ghost = (hit == 0) and ((velocity > 0.01) or (abs(offset) > 0.01))

                if is_ghost:
                    # Plot ghost marker: red X
                    ax.scatter(t, e, s=100, c='red', marker='X', edgecolors='k', linewidths=0.5)
                    continue
                '''

                if hit == 0:
                    continue

                color = interpolate_color(velocity)

                # Offset as marker shape
                if offset < -0.05:
                    marker = '<'
                elif offset > 0.05:
                    marker = '>'
                else:
                    marker = 's'

                ax.scatter(t, e, s=100, c=[color], marker=marker, edgecolors='k', linewidths=0.5)

        # Legend
        legend = [
            mpatches.Patch(facecolor=modern_colors[0], label='Low Velocity'),
            mpatches.Patch(facecolor=modern_colors[1], label='High Velocity'),
            plt.Line2D([0], [0], marker='<', color='w', label='Early', markerfacecolor='gray', markeredgecolor='k', markersize=8),
            plt.Line2D([0], [0], marker='>', color='w', label='Late', markerfacecolor='gray', markeredgecolor='k', markersize=8),
            plt.Line2D([0], [0], marker='s', color='w', label='On-time', markerfacecolor='gray', markeredgecolor='k', markersize=8),
            plt.Line2D([0], [0], marker='X', color='w', label='Ghost (hit=0)', markerfacecolor='red', markeredgecolor='k', markersize=8),
        ]
        ax.legend(handles=legend, loc='upper right', title="Encoding")

        return fig, ax




    #########################################################################################
    # Scratchpad

if __name__ == "__main__":

    import pickle
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_path = os.path.join(project_root, "dataset", "serialized", "80sBlack.pkl")
    with open(test_path, "rb") as f:
        pickle_data = pickle.load(f)

    rand_idx = np.random.randint(0, len(pickle_data))
    print(f"Random index: {rand_idx}")
    sample = pickle_data[rand_idx]
    feature = DrumMIDIFeature(sample["midi_bytes"])
    fixed_grid, _ = feature.to_fixed_grid(steps_per_quarter=4)
    feature_reconstructed = feature.from_fixed_grid(fixed_grid, steps_per_quarter=4)
    feature_reconstructed.play()

    # Simplify
    button_hvo = feature.simplify_to_button_hvo(win_sizes=[(2, 1.0)], velocity_range=(0.5, 0.8), max_hits_per_win=1, win_retain_prob=1.0, steps_per_quarter=4, num_buttons=2)

    # Convert to button HVO
    button_hvo_feature = feature.from_button_hvo(button_hvo, steps_per_quarter=4)
    button_hvo_feature.play_button_hvo(button_hvo_feature)





    '''

    # Reduce
    fixedGrid_reduced = feature.reduce_groove_fixed_grid(velocity_threshold=0.4)
    feature_reconstructed = feature.from_fixed_grid(fixedGrid_reduced, steps_per_quarter=4)
    feature_reconstructed.play()

    '''
