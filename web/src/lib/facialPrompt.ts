import type { FacialTakeSummary } from "../types";
import { formatEmotionTimelineForPrompt } from "./emotionTimelineFormat";

/** Human-readable block for the Gemini prompt (no JSON). */
export function formatFacialSummaryForPrompt(f: FacialTakeSummary | null): string {
  if (!f || f.framesSampled === 0) {
    return "(No facial frames were sampled for this take.)";
  }
  if (f.framesWithFace === 0) {
    return (
      `Face ML ran for ${f.framesSampled} sampled frame(s) but no face was detected reliably. ` +
      `Treat facial signal as unavailable.`
    );
  }

  const sorted = Object.entries(f.emotionCounts).sort((a, b) => b[1] - a[1]);
  const dist = sorted.map(([e, n]) => `${e}: ${n}`).join(", ");
  const snap = f.snapshotBlendshapes
    ? ` Example blendshape keys (0–1): ${Object.entries(f.snapshotBlendshapes)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 6)
        .map(([k, v]) => `${k}=${v.toFixed(2)}`)
        .join(", ")}.`
    : "";

  const timeline = f.emotionTimeline?.length
    ? ` Timeline (seconds from the start of the attached audio take; heuristic face labels, ~same clock as recording): ${formatEmotionTimelineForPrompt(f.emotionTimeline)}.`
    : "";

  return (
    `MediaPipe Face Landmarker → heuristic emotion labels over the recorded take (not clinical, not ground truth). ` +
    `Frames with a detected face: ${f.framesWithFace} / ${f.framesSampled} sampled. ` +
    `Label histogram (count of frames): ${dist}. Dominant: ${f.dominantEmotion ?? "n/a"}.${snap}${timeline} ` +
    `Use timed labels to relate moments in the audio to approximate on-camera expression; prioritize listening; labels are not definitive feelings.`
  );
}
