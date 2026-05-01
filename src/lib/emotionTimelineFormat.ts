import type { EmotionTimelineSample } from "../types";

/** Merge adjacent samples with the same label into readable segments. */
export function formatEmotionTimelineForPrompt(samples: EmotionTimelineSample[]): string {
  if (samples.length === 0) {
    return "(no timed face-emotion samples)";
  }

  const rounded = samples.map((s) => ({
    t: Math.round(s.tSec * 100) / 100,
    e: s.emotion,
  }));

  const segments: { start: number; end: number; e: string }[] = [];
  for (const s of rounded) {
    const last = segments[segments.length - 1];
    if (last && last.e === s.e) {
      last.end = s.t;
    } else {
      segments.push({ start: s.t, end: s.t, e: s.e });
    }
  }

  return segments
    .map((seg) =>
      seg.start === seg.end
        ? `${seg.start.toFixed(2)}s: ${seg.e}`
        : `${seg.start.toFixed(2)}–${seg.end.toFixed(2)}s: ${seg.e}`,
    )
    .join(" · ");
}
