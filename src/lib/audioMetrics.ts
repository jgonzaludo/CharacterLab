import type { VocalMetrics } from "../types";

function clamp01(n: number): number {
  return Math.min(1, Math.max(0, n));
}

function stdDev(values: number[]): number {
  if (values.length === 0) return 0;
  const mean = values.reduce((a, b) => a + b, 0) / values.length;
  const v =
    values.reduce((acc, x) => acc + (x - mean) ** 2, 0) / values.length;
  return Math.sqrt(v);
}

/**
 * Decode recorded media to an AudioBuffer when the browser supports the codec.
 */
export async function decodeAudioFromBlob(blob: Blob): Promise<AudioBuffer | null> {
  const ctx = new AudioContext();
  try {
    const raw = await blob.arrayBuffer();
    return await ctx.decodeAudioData(raw.slice(0));
  } catch {
    return null;
  } finally {
    await ctx.close();
  }
}

/**
 * Heuristic vocal dimensions from amplitude envelopes (not speech recognition).
 * Normalized roughly 0–1 for interpretability alongside LLM commentary.
 */
export function computeVocalMetrics(buffer: AudioBuffer): VocalMetrics {
  const channelData = buffer.getChannelData(0);
  const sampleRate = buffer.sampleRate;
  const chunkSamples = Math.max(256, Math.floor(sampleRate * 0.05));
  const rmsChunks: number[] = [];

  for (let i = 0; i < channelData.length; i += chunkSamples) {
    let sum = 0;
    const end = Math.min(i + chunkSamples, channelData.length);
    for (let j = i; j < end; j++) {
      const s = channelData[j];
      sum += s * s;
    }
    rmsChunks.push(Math.sqrt(sum / (end - i)));
  }

  if (rmsChunks.length === 0) {
    return {
      intensity: 0.5,
      restraint: 0.5,
      tension: 0.5,
      durationSec: buffer.duration,
      metricsApproximate: true,
    };
  }

  const mean = rmsChunks.reduce((a, b) => a + b, 0) / rmsChunks.length;
  const peak = Math.max(...rmsChunks);
  const valley = Math.min(...rmsChunks);
  const range = peak - valley;
  const sd = stdDev(rmsChunks);

  // Intensity: overall energy (scaled; quiet rooms still get usable signal)
  const intensity = clamp01(mean * 4);

  // Restraint: low dynamic swing relative to level → controlled / held-in read
  const swingRatio = range / (mean + 0.02);
  const restraint = clamp01(1 - swingRatio / 3);

  // Tension: relative variability of the envelope
  const tension = clamp01(sd / (mean + 0.02) / 2);

  return {
    intensity,
    restraint,
    tension,
    durationSec: buffer.duration,
    metricsApproximate: false,
  };
}

export async function metricsFromRecording(blob: Blob): Promise<VocalMetrics> {
  const buffer = await decodeAudioFromBlob(blob);
  if (!buffer) {
    return {
      intensity: 0.5,
      restraint: 0.5,
      tension: 0.5,
      durationSec: 0,
      metricsApproximate: true,
    };
  }
  return computeVocalMetrics(buffer);
}
