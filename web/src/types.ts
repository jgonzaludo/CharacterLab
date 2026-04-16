export interface VisionEmotion {
  timestamp: number;
  blendshapes: Record<string, number>;
  primaryEmotion: string;
  confidence: number;
}
