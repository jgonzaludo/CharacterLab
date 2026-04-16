import type { EmotionName } from "./emotionFromBlendshapes";

export const EMOTION_EMOJI: Record<EmotionName, string> = {
  Joy: "😄",
  Sadness: "😢",
  Anger: "😠",
  Surprise: "😲",
  Disgust: "🤢",
  Fear: "😨",
  Neutral: "😐",
};

export const EMOTION_COLOR: Record<EmotionName, string> = {
  Joy: "#fbbf24",
  Sadness: "#60a5fa",
  Anger: "#f87171",
  Surprise: "#a78bfa",
  Disgust: "#34d399",
  Fear: "#fb923c",
  Neutral: "#9ca3af",
};

/** Google-hosted model (same family as local models/face_landmarker.task). */
export const FACE_LANDMARKER_MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

export const MEDIAPIPE_WASM_ROOT = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm`;
