/** A single on/off tap. Extend to an object later for vel/offset if you want. */
export type Tap = 0 | 1;

/**
 * One step is an array of taps, length === tracks.length:
 *   step[t] is 0|1 for track index t
 * Grid is Step[] where grid[stepIndex] => Tap[]
 */
export type Step = Tap[];

/** Helper to build an empty grid (all zeros) */
export function makeEmptyGrid(steps: number, nTracks: number): Step[] {
  return Array.from({ length: steps }, () => Array(nTracks).fill(0 as Tap));
}

export type DrumClass = 'kick' | 'snare' | 'hhc' | 'hho' | 'ltom' | 'mtom' | 'htom' | 'crash' | 'ride';
export type HitClass  = 'hit0' | 'hit1';

export type HitEvent = {
    inst: HitClass;
    t: number;   // 0 - 32 (1/16nd note)
}

export type DrumEvent = {
    inst: DrumClass;
    t: number;   // 0 - 32 (1/16nd note)
    hit: number; // 0-1
    vel: number; // 0-1
    offset: number; // -0.5 to 0.5
}

export type GenerateRequest = {
    hits: HitEvent[];
}

export type GenerateResponse = {
    drums: DrumEvent[];
}

