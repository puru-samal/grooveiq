import * as React from "react";
import {
  Box,
  Card,
  CardHeader,
  CardContent,
  useTheme,
  Stack,
  Tooltip,
  IconButton,
  Divider,
} from "@mui/material";
import {
  FiberManualRecordRounded as RecordIcon,
  PlayArrowRounded as PlayIcon,
  StopRounded as StopIcon,
  DeleteSweepRounded as ClearIcon,
} from "@mui/icons-material";
import { type Step, makeEmptyGrid } from "./utils";

export interface PianoRollProps {
  /** Ticks per quarter note (subdivisions inside a quarter). 4=16ths across a bar of 4/4 if qpb=4. */
  tpq?: number;
  /** Quarter notes per bar (time signature numerator for a /4 meter). */
  qpb?: number;
  /** Total bars. */
  bars?: number;
  /** Track labels (top→bottom order). Required to know how many lanes to draw. */
  tracks: string[];

  /** Controlled grid; if omitted, component manages its own internal state. */
  value?: Step[];
  /** onChange fires with the next grid when a user toggles a cell. */
  onChange?: (next: Step[]) => void;

  /** Optional moving cursor index (0..steps-1). Draws a vertical playhead line. */
  cursorIndex?: number;

  /** Draw a stronger vertical divider at the start of each beat (every tpq). */
  showBeatDividers?: boolean;

  /** Disable interactions. */
  disabled?: boolean;

  /** Optional title/subheader for the Card header. */
  title?: string;
  subheader?: string;

  /** Controls how much space the grid takes up */
  size?: "small" | "medium" | "large" | "full";

  /** Transport callbacks + state (controlled by parent) */
  onRecord?: () => void;
  onPlay?: () => void;
  onStop?: () => void;
  onClear?: () => void;
  isRecording?: boolean;
  isPlaying?: boolean;
}

/**
 * Fully dynamic, multi-lane piano roll rendered with SVG.
 * - Transport bar (record / play / stop / clear)
 * - Background lanes, beat/step dividers, step numbers
 * - Rounded "note" rectangles for active cells
 * - Transparent hit areas for click-to-toggle
 * - Optional cursor playhead
 */
export default function PianoRoll({
  tpq = 4,
  qpb = 4,
  bars = 2,
  tracks,
  value,
  onChange,
  cursorIndex,
  size = "medium",
  showBeatDividers = true,
  disabled = false,
  title = "Piano Roll",
  subheader,

  // transport
  onRecord,
  onPlay,
  onStop,
  onClear,
  isRecording = false,
  isPlaying = false,
}: PianoRollProps) {
  const theme = useTheme();

  // --- derived sizes ---
  const nTracks = tracks.length;
  const steps = bars * qpb * tpq; // total columns

  // --- internal vs controlled state ---
  const [internal, setInternal] = React.useState<Step[]>(
    value ?? makeEmptyGrid(steps, nTracks),
  );
  const grid = value ?? internal; // source of truth for rendering

  // Keep grid shape in sync if steps or number of tracks change
  React.useEffect(() => {
    const target = makeEmptyGrid(steps, nTracks);
    for (let i = 0; i < Math.min(grid.length, steps); i++) {
      for (let t = 0; t < Math.min(grid[i]?.length ?? 0, nTracks); t++) {
        target[i][t] = grid[i][t];
      }
    }
    if (value && onChange) onChange(target);
    else setInternal(target);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [steps, nTracks]);

  // Toggle a single cell (trackIndex, stepIndex)
  const toggleCell = (tIndex: number, sIndex: number) => {
    if (disabled) return;
    const next = grid.map((row) => row.slice()); // shallow copy 2D
    next[sIndex][tIndex] = next[sIndex][tIndex] ? 0 : 1;
    if (onChange) onChange(next);
    else setInternal(next);
  };

  // --- layout math for SVG ---
  const cols = steps;
  const rows = nTracks;

  // responsive sizing
  const getSizeConfig = () => {
    switch (size) {
      case "small":
        return { cellW: 20, cellH: 18, gapX: 1, gapY: 4 };
      case "large":
        return { cellW: 32, cellH: 28, gapX: 3, gapY: 8 };
      case "full":
        return { cellW: 40, cellH: 36, gapX: 4, gapY: 10 };
      case "medium":
      default:
        return { cellW: 26, cellH: 22, gapX: 2, gapY: 6 };
    }
  };
  const {
    cellW: effectiveCellW,
    cellH: effectiveCellH,
    gapX: effectiveGapX,
    gapY: effectiveGapY,
  } = getSizeConfig();

  const colPitch = effectiveCellW + effectiveGapX;
  const rowPitch = effectiveCellH + effectiveGapY;

  const svgW = cols * colPitch - effectiveGapX;
  const svgH = rows * rowPitch - effectiveGapY;

  const xForCol = (i: number) => i * colPitch;
  const yForRow = (r: number) => r * rowPitch;

  const titleId = React.useId();
  const descId = React.useId();

  // Small palette: cycles through a few theme colors for lanes
  const laneColor = (r: number) => {
    const palette = [
      theme.palette.primary.main,
      theme.palette.success.main,
      theme.palette.info.main,
      theme.palette.warning.main,
      theme.palette.secondary.main,
    ];
    return palette[r % palette.length];
  };

  // --- render helpers ---

  const renderBackground = () => (
    <g>
      {/* Lane backgrounds */}
      {tracks.map((_, r) => (
        <rect
          key={`lane-${r}`}
          x={0}
          y={yForRow(r)}
          width={svgW}
          height={effectiveCellH}
          rx={6}
          ry={6}
          fill={
            theme.palette.mode === "dark"
              ? "rgba(255,255,255,0.04)"
              : "rgba(0,0,0,0.04)"
          }
        />
      ))}

      {/* Vertical grid lines; strong at each beat (every tpq steps) */}
      {Array.from({ length: cols }, (_, i) => {
        const x = xForCol(i);
        const isBeat = i % tpq === 0;
        return (
          <line
            key={`vl-${i}`}
            x1={x}
            y1={0}
            x2={x}
            y2={svgH}
            stroke={
              showBeatDividers
                ? isBeat
                  ? theme.palette.divider
                  : theme.palette.mode === "dark"
                  ? "rgba(255,255,255,0.08)"
                  : "rgba(0,0,0,0.06)"
                : theme.palette.mode === "dark"
                ? "rgba(255,255,255,0.06)"
                : "rgba(0,0,0,0.05)"
            }
            strokeWidth={showBeatDividers && isBeat ? 1.25 : 0.75}
          />
        );
      })}

      {/* Step numbers, top and bottom */}
      {Array.from({ length: cols }, (_, i) => {
        const x = xForCol(i) + effectiveCellW / 2;
        const isDownbeat = i % tpq === 0;
        return (
          <text
            key={`num-top-${i}`}
            x={x}
            y={-6}
            textAnchor="middle"
            fontSize={10}
            fill={isDownbeat ? theme.palette.text.primary : theme.palette.text.secondary}
            opacity={isDownbeat ? 0.95 : 0.75}
          >
            {i + 1}
          </text>
        );
      })}
      {Array.from({ length: cols }, (_, i) => {
        const x = xForCol(i) + effectiveCellW / 2;
        const isDownbeat = i % tpq === 0;
        return (
          <text
            key={`num-bot-${i}`}
            x={x}
            y={svgH + 14}
            textAnchor="middle"
            fontSize={10}
            fill={isDownbeat ? theme.palette.text.primary : theme.palette.text.secondary}
            opacity={isDownbeat ? 0.95 : 0.75}
          >
            {i + 1}
          </text>
        );
      })}
    </g>
  );

  const renderCursor = () => {
    if (cursorIndex === undefined) return null;
    const x = xForCol(cursorIndex);
    return (
      <g>
        <line
          x1={x}
          y1={-10}
          x2={x}
          y2={svgH + 10}
          stroke={theme.palette.warning.main}
          strokeWidth={2}
        />
      </g>
    );
  };

  const renderNotes = () => (
    <g>
      {tracks.map((trackName, r) =>
        grid.map((step, i) => {
          if (!step[r]) return null;
          const x = xForCol(i) + 1;
          const y = yForRow(r) + 3;
          const color = laneColor(r);
          return (
            <g key={`${r}-${i}`}>
              <title>{`${trackName} • Step ${i + 1}`}</title>
              <rect
                x={x}
                y={y}
                width={effectiveCellW - 2}
                height={effectiveCellH - 6}
                rx={5}
                ry={5}
                fill={color}
                opacity={0.95}
                filter="url(#shadowSoft)"
              />
            </g>
          );
        }),
      )}
    </g>
  );

  const renderHitAreas = () => (
    <g>
      {tracks.map((_, r) =>
        Array.from({ length: cols }, (_, i) => {
          const x = xForCol(i);
          const y = yForRow(r);
          return (
            <rect
              key={`hit-${r}-${i}`}
              x={x}
              y={y}
              width={effectiveCellW}
              height={effectiveCellH}
              fill="transparent"
              style={{ cursor: disabled ? "default" : "pointer" }}
              onClick={() => toggleCell(r, i)}
            />
          );
        }),
      )}
    </g>
  );

  // --- simple mini transport bar ---
  const TransportBar = () => (
    <Stack
      direction="row"
      alignItems="center"
      justifyContent="center"
      spacing={1}
      sx={{
        mb: 1,
        px: 1,
        py: 0.5,
        borderRadius: 1,
        bgcolor:
          theme.palette.mode === "dark"
            ? "rgba(255,255,255,0.04)"
            : "rgba(0,0,0,0.035)",
        width: "fit-content",
        mx: "auto",
      }}
    >
      <Tooltip title={isRecording ? "Stop Recording" : "Record"}>
        <span>
          <IconButton
            size="small"
            onClick={onRecord}
            disabled={disabled}
            sx={{
              color: isRecording ? theme.palette.error.main : theme.palette.text.secondary,
            }}
          >
            <RecordIcon fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>

      <Divider flexItem orientation="vertical" sx={{ mx: 0.5, opacity: 0.3 }} />

      {/* Toggleable Play/Stop Button */}
      <Tooltip title={isPlaying ? "Stop" : "Play"}>
        <span>
          <IconButton
            size="small"
            onClick={isPlaying ? onStop : onPlay}
            disabled={disabled}
            sx={{
              color: isPlaying ? theme.palette.success.main : theme.palette.text.secondary,
            }}
          >
            {isPlaying ? <StopIcon fontSize="small" /> : <PlayIcon fontSize="small" />}
          </IconButton>
        </span>
      </Tooltip>

      <Divider flexItem orientation="vertical" sx={{ mx: 0.5, opacity: 0.3 }} />

      <Tooltip title="Clear">
        <span>
          <IconButton size="small" onClick={onClear} disabled={disabled}>
            <ClearIcon fontSize="small" />
          </IconButton>
        </span>
      </Tooltip>
    </Stack>
  );

  return (
    <Card
      variant="outlined"
      sx={{
        borderRadius: 2,
        boxShadow:
          theme.palette.mode === "dark"
            ? "0 4px 20px rgba(0,0,0,0.35)"
            : "0 6px 24px rgba(0,0,0,0.10)",
      }}
    >
      <CardHeader
        title={title}
        subheader={
          subheader ?? `${steps} steps • ${rows} lanes • tpq=${tpq}, qpb=${qpb}, bars=${bars}`
        }
        sx={{
          "& .MuiCardHeader-title": { fontWeight: 700 },
          "& .MuiCardHeader-subheader": { opacity: 0.8 },
          pb: 0.5,
        }}
      />

      <CardContent>
        {/* Mini transport bar */}
        <TransportBar />

        <Box
          sx={{
            position: "relative",
            overflowX: "auto",
            pt: 2.5, // top padding for top numbers
            pb: 3,   // bottom padding for bottom numbers
          }}
        >
          <svg
            width={svgW}
            height={svgH + 20}
            viewBox={`0 -14 ${svgW} ${svgH + 30}`}
            role="img"
            aria-labelledby={titleId}
            aria-describedby={descId}
          >
            <title id={titleId}>Multi-lane Piano Roll Grid</title>
            <desc id={descId}>
              {rows} lanes and {cols} columns. Click a cell to toggle a note.
            </desc>

            {/* soft shadow for note blocks */}
            <defs>
              <filter id="shadowSoft" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="0" dy="1" stdDeviation="1" floodOpacity="0.25" />
              </filter>
            </defs>

            {renderBackground()}
            {renderCursor()}
            {renderNotes()}
            {renderHitAreas()}
          </svg>
        </Box>
      </CardContent>
    </Card>
  );
}
