import curses
import time
import mido
from mido import Message, MidiFile, MidiTrack
import io
import numpy as np
import torch
import yaml
import os
from data import DrumMIDIFeature, CANONICAL_DRUM_MAP, DrumMIDIDataset
from models import GrooveIQ
import pygame
from typing import Optional
from tqdm import tqdm

# ========== Config ==========
KEY_NOTE_MAP = {ord('a'): 36, ord('l'): 38}
BPM = 80
VELOCITY = 120
CHANNEL = 9
DURATION = 0.01
BEATS_PER_BAR = 4
NUM_BARS = 2
COUNT_IN_BEATS = 4
RECORD_SECONDS = 60.0 / BPM * BEATS_PER_BAR * NUM_BARS
OUTPUT_FILE = "recording.mid"

# Model + Mapping setup
MAX_LENGTH = 33
E = 9
M = 3
fixed_grid_drum_mapping = {pitch: [i] for i, pitch in enumerate(CANONICAL_DRUM_MAP.keys())}
device = "cuda" if torch.cuda.is_available() else "cpu"

# ========== Load Model ==========
expt_path = "expts/giq_exp5_heur_causal"
config_path = os.path.join(expt_path, "config.yaml")
chkpt_path = os.path.join(expt_path, "checkpoints", "checkpoint-ep4-model.pth")
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

model_config = config["model"]
model_config.update(T=MAX_LENGTH, E=E, M=M)
model = GrooveIQ(**model_config)
checkpoint = torch.load(chkpt_path, map_location=device, weights_only=True)
print(f"Loading model from {chkpt_path}: ", end="")
model.load_state_dict(checkpoint['model_state_dict'], strict=True)
model.to(device)
model.eval()

# ========== Load Style Set ==========
STYLE_SET_PATH = os.path.join(expt_path, "style_set.pkl")
style_set = DrumMIDIDataset(
    path     = STYLE_SET_PATH,
    num_bars = config["data"]["num_bars"],
    feature_type = config["data"]["feature_type"],
    steps_per_quarter = config["data"]["steps_per_quarter"],
    subset   = 1.0,
    aug_config = None,
    calc_desc = False
)

# ========== Collect Style Samples ==========
style_map = {style: [] for style in style_set.data_stats.style_map.keys() if style != "unknown"}
errors    = 0
pbar = tqdm(total=len(style_set), desc="Collecting Encoded Vectors")
for i in range(len(style_set)):
    try:
        sample, _, _, _ = style_set[i]
        if sample.style == "unknown":
            pbar.update(1)
            continue
        style_map[sample.style].append(i)
        pbar.update(1)
    except Exception as e:
        print(f"Error collecting style samples {i}: {e}")
        errors += 1
        pbar.update(1)
        continue

pbar.close()
print(f"Total errors: {errors}")



# ========== Audio Setup ==========
pygame.init()
pygame.mixer.init()

def make_click(freq=1000, duration=0.05, volume=0.5, sample_rate=44100):
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    waveform = (np.sin(2 * np.pi * freq * t) * volume * 32767).astype(np.int16)
    stereo = np.stack([waveform, waveform], axis=-1)
    return pygame.sndarray.make_sound(stereo)

click_sound = make_click()


def quantize_event_buffer(event_buffer, bpm=120, division=32):
    if not event_buffer:
        return []
    min_time = min(e[2] for e in event_buffer)
    beat_duration = 60.0 / bpm
    grid_interval = beat_duration / (division / 4)
    return [(kind, note, round((t - min_time) / grid_interval) * grid_interval, vel)
            for kind, note, t, vel in event_buffer]


def write_midi(events, filename=OUTPUT_FILE):
    mid = MidiFile()
    track = MidiTrack()
    mid.tracks.append(track)

    tempo = mido.bpm2tempo(BPM)
    track.append(mido.MetaMessage('set_tempo', tempo=tempo))
    track.append(mido.MetaMessage('time_signature', numerator=4, denominator=4,
                                  clocks_per_click=24, notated_32nd_notes_per_beat=8))

    full_events = sorted(events, key=lambda x: x[2])
    last_tick = 0
    for kind, note, t, vel in full_events:
        tick = mido.second2tick(t, mid.ticks_per_beat, tempo)
        track.append(Message('note_on', note=note, velocity=vel, time=int(tick - last_tick), channel=CHANNEL))
        track.append(Message('note_off', note=note, velocity=vel, time=int(DURATION * mid.ticks_per_beat), channel=CHANNEL))
        last_tick = tick

    mid.save(filename)
    with io.BytesIO() as buf:
        mid.save(file=buf)
        return buf.getvalue()
    

def safe_addstr(stdscr, text: str):
    """Adds a string to the screen, scrolling if needed."""
    max_y, _ = stdscr.getmaxyx()
    cur_y, _ = stdscr.getyx()
    if cur_y >= max_y - 1:
        stdscr.scroll()
        stdscr.move(max_y - 2, 0)
    stdscr.addstr(text)
    stdscr.refresh()


def record_sequence(stdscr):
    curses.noecho()
    stdscr.nodelay(True)
    stdscr.clear()
    stdscr.addstr("Count-in:\n")
    stdscr.refresh()

    for i in range(COUNT_IN_BEATS):
        click_sound.play()
        safe_addstr(stdscr, f"Click {i+1}\n")
        time.sleep(60.0 / BPM)

    safe_addstr(stdscr, "Recording now! Press A or L...\n")
    

    start_time = time.time()
    event_buffer = []

    while time.time() - start_time < RECORD_SECONDS:
        ch = stdscr.getch()
        if ch in KEY_NOTE_MAP:
            note = KEY_NOTE_MAP[ch]
            t = time.time() - start_time
            event_buffer.append(("note_on", note, t, VELOCITY))
            
            safe_addstr(stdscr, f"Key {chr(ch)} -> note {note} at {t:.3f}s\n")
        
        time.sleep(0.01)

    safe_addstr(stdscr, "\nRecording complete. Running model...\n")
    return quantize_event_buffer(event_buffer, bpm=BPM, division=16)

def inference(model : GrooveIQ, button_hits : torch.Tensor, grid : Optional[torch.Tensor] = None, device : str = "cpu", threshold : float = 0.85):
    model.to(device)
    model.eval()
    with torch.no_grad():
        # Encode grid if provided
        encoded, button_repr = None, None
        if grid is not None:
            encoded, button_repr = model.encode(grid)

        # Make button_embed from either user-provided button_hits or learned button_repr
        # depending on the model's configuration
        # If both are available, button_hits is used
        if button_hits is None:
            if button_repr is None:
                raise ValueError("button_repr is None. Either provide button_hits or grid.")
            button_hits = model.make_button_hits(button_repr)
            
        button_embed = model.make_button_embed(button_hits)
        if encoded is None:
            z, _, _ = model.make_z_prior(button_embed)
        else:
            z, _, _ = model.make_z_post(button_embed, encoded)
        
        generated_grid, _ = model.generate(button_embed, z, max_steps=MAX_LENGTH, threshold=threshold)
        generated_grid = generated_grid[:, 1:, :, :] # Drop SOS token
    return generated_grid

def main_loop(stdscr):
    curses.curs_set(0)           # Hide cursor (optional)
    stdscr.scrollok(True)
    stdscr.idlok(True)
    while True:
        events = record_sequence(stdscr)
        midi_bytes = write_midi(events)

        # Process Control Sequence
        feature = DrumMIDIFeature(midi_bytes)
        button_hvo = feature.to_button_hvo(steps_per_quarter=4, num_buttons=2)
        button_feature = feature.from_button_hvo(button_hvo, steps_per_quarter=4)
        button_hits = button_hvo[:, :, 0].unsqueeze(0).to(device)

        def generate_and_play(grid=None, style_name=None):
            gen_grid = inference(model, button_hits, grid, device=device, threshold=0.85)
            out = feature.from_fixed_grid(gen_grid.squeeze(0).detach().cpu(), steps_per_quarter=4)
            if style_name:
                safe_addstr(stdscr, f"\nGenerated with style: {style_name}\n")
            else:
                safe_addstr(stdscr, f"\nGenerated with style: [None]\n")
            out.play()

        # 1st generation without style
        generate_and_play(grid=None, style_name=None)

        safe_addstr(stdscr,
            "\n[R] Re-record\n"
            "[P] Replay button sequence\n"
            "[S] Sample another style\n"
            "[N] Generate with no style\n"
            "[Q] Quit\n"
        )

        while True:
            ch = stdscr.getch()
            if ch == ord('q'):
                return
            elif ch == ord('r'):
                stdscr.clear()
                break
            elif ch == ord('p'):
                button_feature.play_button_hvo(button_feature)
            elif ch == ord('s'):
                new_style_idx = np.random.choice(list(style_map.keys()))
                new_sample_idx = np.random.choice(style_map[new_style_idx])
                _, new_grid, _, _ = style_set[new_sample_idx]
                generate_and_play(grid=new_grid.unsqueeze(0), style_name=new_style_idx)
            elif ch == ord('n'):
                generate_and_play(grid=None, style_name=None)

            time.sleep(0.1)


if __name__ == "__main__":
    curses.wrapper(main_loop)
