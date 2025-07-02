try:
    import fluidsynth
    _HAS_FLUIDSYNTH = True
except ImportError:
    _HAS_FLUIDSYNTH = False
import os
import numpy as np
from symusic import Score, TimeUnit
import math
import torch


def get_num_bars(duration: int, time_signature: tuple[int, int], tpq: int) -> float:
    """
    Compute the number of bars in the score.

    Args:
        duration: The duration of the MIDI file in ticks.
        time_signature: The time signature of the MIDI file in the format (beats, beat_resolution).
        tpq: The ticks per quarter note of the MIDI file.

    Returns:
        float: Number of bars.
    """
    beats_per_bar = time_signature[0]
    beat_unit = time_signature[1]
    ticks_per_beat = tpq * (4 / beat_unit)
    ticks_per_bar = beats_per_bar * ticks_per_beat
    return math.ceil(duration / ticks_per_bar)


def get_num_quarters(duration: int, time_signature: tuple[int, int], tpq: int) -> int:
    """
    Compute the number of quarter notes in the score, ceiling-rounded to the nearest bar.

    Args:
        duration: The duration of the MIDI file in ticks.
        time_signature: The time signature in (beats per bar, beat unit).
        tpq: Ticks per quarter note.

    Returns:
        int: Total number of quarter notes, ceiling-rounded to the next full bar.
    """
    beats_per_bar = time_signature[0]
    beat_unit = time_signature[1]

    # Ticks per bar
    ticks_per_beat = tpq * (4 / beat_unit)
    ticks_per_bar = beats_per_bar * ticks_per_beat

    # Total bars (ceiled)
    num_bars = math.ceil(duration / ticks_per_bar)

    # Quarters per bar
    quarters_per_bar = beats_per_bar * (4 / beat_unit)

    # Total quarters (rounded to full bars)
    return int(num_bars * quarters_per_bar)


def note_to_ticks(note_type: str, ticks_per_quarter: int) -> int:
    """
    Convert a musical note type into tick count based on the given PPQN (ticks per quarter note).
    
    Args:
        note_type (str): Note duration as a string (e.g., 'quarter', 'eighth', 'dotted_eighth', 'triplet_sixteenth').
        ticks_per_quarter (int): Ticks per quarter note (tpq) value.

    Returns:
        int: Number of ticks corresponding to the note_type.
    """
    duration_ratios = {
        'whole': 4,
        'half': 2,
        'quarter': 1,
        'eighth': 1/2,
        'sixteenth': 1/4,
        'thirty_second': 1/8,
        'sixty_fourth': 1/16,
        'dotted_half': 3,
        'dotted_quarter': 1.5,
        'dotted_eighth': 0.75,
        'dotted_sixteenth': 0.375,
        'triplet_half': 4/3,
        'triplet_quarter': 2/3,
        'triplet_eighth': 1/3,
        'triplet_sixteenth': 1/6,
        'triplet_thirty_second': 1/12
    }
    
    ratio = duration_ratios.get(note_type.lower())
    if ratio is None:
        raise ValueError(f"Unknown note type: {note_type}")
    
    return round(ratio * ticks_per_quarter)


def render(score: Score, sf_path: str, fs: int = 44100) -> np.ndarray:
    """
    Synthesize the score to a stereo waveform using FluidSynth.
    
    Args:
        score: The score object.
        sf_path: The path to the soundfont file.
        fs: Sample rate for synthesis.

    Returns:
        np.ndarray: Stereo audio waveform (shape: [samples, 2]).

    Raises:
        ImportError: If FluidSynth is not installed.
        ValueError: If the SoundFont file is missing.
    """
    score_s = score.to(TimeUnit.second)

    if not _HAS_FLUIDSYNTH:
        raise ImportError("render() was called but pyfluidsynth is not installed.")

    if not os.path.exists(sf_path):
        raise ValueError(f"No soundfont file found at the supplied path {sf_path}")

    if len(score_s.tracks) == 0 or all(len(i.notes) == 0 for i in score_s.tracks):
        return np.array([])

    waveforms: list[np.ndarray] = []
    for track in score_s.tracks:
        if len(track.notes) == 0:
            continue

        fl = fluidsynth.Synth(samplerate=fs)
        sfid = fl.sfload(sf_path)

        if track.is_drum:
            channel = 9
            res = fl.program_select(channel, sfid, 128, track.program)
            if res == -1:
                fl.program_select(channel, sfid, 128, 0)
        else:
            channel = 0
            fl.program_select(channel, sfid, 0, track.program)

        event_list = []
        for note in track.notes:
            event_list += [[note.time, 'note on', note.pitch, note.velocity]]
            event_list += [[note.time + note.duration, 'note off', note.pitch]]
        for bend in track.pitch_bends:
            event_list += [[bend.time, 'pitch bend', bend.value]]
        for control_change in track.controls:
            event_list += [[control_change.time, 'control change',
                            control_change.number, control_change.value]]
        event_list.sort(key=lambda x: (x[0], x[1] != 'note off'))
        current_time = event_list[0][0]
        next_event_times = [e[0] for e in event_list[1:]]
        for event, end in zip(event_list[:-1], next_event_times):
            event[0] = end - event[0]
        event_list[-1][0] = 1.
        total_time = current_time + np.sum([e[0] for e in event_list])
        synthesized = np.zeros((int(np.ceil(fs * total_time)), 2))
        for event in event_list:
            if event[1] == 'note on':
                fl.noteon(channel, event[2], event[3])
            elif event[1] == 'note off':
                fl.noteoff(channel, event[2])
            elif event[1] == 'pitch bend':
                fl.pitch_bend(channel, event[2])
            elif event[1] == 'control change':
                fl.cc(channel, event[2], event[3])
            current_sample = int(fs * current_time)
            end = int(fs * (current_time + event[0]))
            samples = fl.get_samples(end - current_sample)
            samples_left = samples[::2]
            samples_right = samples[1::2]
            assert len(samples_left) == len(samples_right)
            synthesized[current_sample:end, 0] += samples_left
            synthesized[current_sample:end, 1] += samples_right
            current_time += event[0]
        fl.delete()
        waveforms.append(synthesized)

    if not waveforms:
        return np.array([])

    max_len = max(w.shape[0] for w in waveforms)
    synthesized = np.zeros((max_len, 2))
    for waveform in waveforms:
        synthesized[:waveform.shape[0]] += waveform

    max_abs = np.abs(synthesized).max()
    if max_abs > 1e-8:
        synthesized /= max_abs

    return synthesized