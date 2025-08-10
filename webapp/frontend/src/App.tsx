// App.tsx
import "./App.css";
import * as React from "react";
import { Container, Stack, Button, TextField } from "@mui/material";
import { makeEmptyGrid, type Step } from "./utils";
// ✅ Import the dynamic piano roll
import PianoRoll from "./PianoRoll";

export default function App() {
  /**
   * --- GRID/TRANSPORT SETTINGS ---
   * tpq = ticks per quarter note (4 = 16th-note grid when qpb=4)
   * qpb = quarter notes per bar (4 = 4/4)
   * bars = number of bars
   */
  const tpq = 4;
  const qpb = 4;
  const bars = 2;
  const steps = tpq * qpb * bars; // = 32 by default

  /**
   * Define tracks (top → bottom order). Add more if you like, the roll scales.
   * If you only want 2 hits, leave it as two labels.
   */
  const tracks = ["Hit 1", "Hit 2"];

  /**
   * --- STATE ---
   * grid: Step[] where Step = Tap[] and Tap = 0|1. Each Step has length = tracks.length.
   * bpm: tempo for cursor timing
   * cursor: moving playhead column; undefined hides it
   */
  const [grid, setGrid] = React.useState<Step[]>(
    makeEmptyGrid(steps, tracks.length),
  );
  const [bpm, setBpm] = React.useState(96);
  const [cursor, setCursor] = React.useState<number | undefined>(0);

  /**
   * --- PLAYBACK LOOP ---
   * Advise: later switch to Tone.js Transport for sample-accurate timing.
   */
  const timerRef = React.useRef<number | null>(null);
  const sixteenthMs = React.useMemo(() => 60000 / bpm / 4, [bpm]);

  const handlePlay = () => {
    setCursor((c) => (typeof c === "number" ? c : 0));
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    timerRef.current = window.setInterval(() => {
      setCursor((c) => (typeof c === "number" ? (c + 1) % steps : 0));
    }, sixteenthMs);
  };

  const handleStop = () => {
    if (timerRef.current) {
      window.clearInterval(timerRef.current);
      timerRef.current = null;
    }
    setCursor(undefined);
  };

  const handleRecord = () => {
    // Show playhead but don't advance it automatically.
    setCursor((c) => (typeof c === "number" ? c : 0));
  };

  const handleClear = () => {
    handleStop();
    setGrid(makeEmptyGrid(steps, tracks.length));
  };

  /**
   * --- DATA UTILITIES ---
   * 1) toChannels: return per-track binary arrays (object keyed by track label)
   * 2) toMergedTwoTrack: for exactly 2 tracks, pack into 0/1/2/3 (A=1, B=2)
   */
  const toChannels = (g: Step[], trackLabels: string[]) =>
    Object.fromEntries(
      trackLabels.map((label, idx) => [label, g.map((step) => step[idx])]),
    );

  const toMergedTwoTrack = (g: Step[]) => {
    if (tracks.length !== 2) {
      throw new Error("toMergedTwoTrack expects exactly 2 tracks.");
    }
    return g.map((step) => (step[0] ? 1 : 0) + (step[1] ? 2 : 0));
  };

  const handleGenerate = async () => {
    // Choose the encoding your backend/model expects:
    const payload = {
      tempo: bpm,
      steps,
      // channels: toChannels(grid, tracks),     // per-track 0/1 arrays
      mergedInputs: toMergedTwoTrack(grid),     // 0/1/2/3 encoding (two tracks only)
      tracks,
      tpq,
      qpb,
      bars,
    };
    console.log("Generate payload →", payload);

    // Example axios call (uncomment when backend is ready):
    // const { data } = await axios.post(
    //   `${import.meta.env.VITE_API_URL}/api/generate`,
    //   payload
    // );
    // console.log("Model response:", data);
  };

  const handleSave = () => {
    const blob = new Blob([JSON.stringify({ grid, tracks, tpq, qpb, bars }, null, 2)], {
      type: "application/json",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "groove_inputs.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  React.useEffect(() => {
    return () => {
      if (timerRef.current) window.clearInterval(timerRef.current);
    };
  }, []);

  return (
    <>
      {/* Simple header; your CSS can style this however you want */}
      <div className="Header">
        <h1>GrooveIQ Demo</h1>
        <p>Multi-lane piano roll. Click to toggle steps, play to move the cursor.</p>
        <div className="Header-buttons">
          <button onClick={handleRecord}>Record</button>
          <button onClick={handlePlay}>Play</button>
          <button onClick={handleStop}>Stop</button>
          <button onClick={handleClear}>Clear</button>
          <button onClick={handleGenerate}>Generate</button>
          <button onClick={handleSave}>Save</button>
        </div>
      </div>

      <Container sx={{ py: 4 }}>
        <Stack spacing={2}>
          {/* Tempo + quick cursor toggle */}
          <Stack direction="row" spacing={2} alignItems="center">
            <TextField
              size="small"
              label="Tempo (BPM)"
              type="number"
              value={bpm}
              onChange={(e) =>
                setBpm(Math.max(40, Math.min(220, Number(e.target.value) || 96)))
              }
              sx={{ width: 150 }}
            />
            <Button
              variant={cursor === undefined ? "outlined" : "contained"}
              onClick={() => setCursor(cursor === undefined ? 0 : undefined)}
            >
              {cursor === undefined ? "Show Cursor" : "Hide Cursor"}
            </Button>
          </Stack>

          {/* 🎹 The dynamic piano roll (controlled) */}
          <PianoRoll
            tpq={tpq}
            qpb={qpb}
            bars={bars}
            tracks={tracks}
            value={grid}
            onChange={setGrid}
            cursorIndex={cursor}
            size="medium"
            showBeatDividers
            title="Hits"
            subheader="Click to toggle steps, play to move the cursor."
            onRecord={handleRecord}
            onPlay={handlePlay}
            onStop={handleStop}
            onClear={handleClear}
            isPlaying={cursor !== undefined}
          />
        </Stack>
      </Container>
    </>
  );
}
