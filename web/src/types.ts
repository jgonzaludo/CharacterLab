/** One heuristic face-emotion label at a moment in the take (seconds from audio start). */
export interface EmotionTimelineSample {
  /** Seconds from the start of the recorded take (aligned with the audio clip). */
  tSec: number;
  emotion: string;
}

/** Aggregated MediaPipe face signals during a single recorded take (heuristic, not clinical). */
export interface FacialTakeSummary {
  emotionCounts: Record<string, number>;
  framesSampled: number;
  framesWithFace: number;
  dominantEmotion: string | null;
  /** Downsampled sequence of face-emotion labels vs time for alignment with audio. */
  emotionTimeline?: EmotionTimelineSample[];
  /** Last frame’s blendshape scores when a face was visible (optional context for the model). */
  snapshotBlendshapes?: Record<string, number>;
  /** True when no face was tracked or nothing was sampled. */
  approximate: boolean;
}

export interface VocalMetrics {
  intensity: number;
  restraint: number;
  tension: number;
  durationSec: number;
  /** True when Web Audio could not decode the take (metrics are neutral placeholders). */
  metricsApproximate: boolean;
}

export interface RehearsalSuggestion {
  id: string;
  title: string;
  body: string;
  reasoning: string;
  tryThis: string;
}

export interface RehearsalFeedbackResult {
  howItMayRead: string;
  interpretationNotes: string;
  dimensionNotes?: {
    intensity?: string;
    restraint?: string;
    tension?: string;
  };
  suggestions: RehearsalSuggestion[];
  approximateTranscript?: string;
}

export type SuggestionRating = "up" | "down" | null;
