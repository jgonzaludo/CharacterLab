/**
 * Heuristic mapping of MediaPipe blendshapes to discrete emotions.
 * Aligned with core/vision_engine.py — keep in sync with Python.
 */
export type EmotionName =
  | "Joy"
  | "Sadness"
  | "Anger"
  | "Surprise"
  | "Disgust"
  | "Fear"
  | "Neutral";

export interface EmotionResult {
  primaryEmotion: EmotionName;
  confidence: number;
}

export function mapBlendshapesToEmotion(bs: Record<string, number>): EmotionResult {
  const scores: Record<Exclude<EmotionName, "Neutral">, number> = {
    Joy: ((bs.mouthSmileLeft ?? 0) + (bs.mouthSmileRight ?? 0)) / 2,
    Sadness:
      ((bs.browInnerUp ?? 0) + (bs.mouthFrownLeft ?? 0) + (bs.mouthFrownRight ?? 0)) / 3,
    Anger:
      ((bs.browDownLeft ?? 0) +
        (bs.browDownRight ?? 0) +
        (bs.eyeSquintLeft ?? 0) +
        (bs.eyeSquintRight ?? 0)) /
      3.2,
    Surprise:
      ((bs.browOuterUpLeft ?? 0) +
        (bs.browOuterUpRight ?? 0) +
        (bs.eyeWideLeft ?? 0) +
        (bs.eyeWideRight ?? 0) +
        (bs.jawOpen ?? 0)) /
      4.5,
    Disgust: ((bs.noseSneerLeft ?? 0) + (bs.noseSneerRight ?? 0)) / 2,
    Fear:
      ((bs.browInnerUp ?? 0) + (bs.eyeWideLeft ?? 0) + (bs.eyeWideRight ?? 0)) / 3,
  };

  const primary = (
    Object.keys(scores) as (keyof typeof scores)[]
  ).reduce((a, b) => (scores[a] >= scores[b] ? a : b));

  const confidence = scores[primary];

  if (confidence < 0.2) {
    return { primaryEmotion: "Neutral", confidence: 1 };
  }

  return { primaryEmotion: primary, confidence };
}
