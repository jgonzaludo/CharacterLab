import { useEffect, useRef, useState, type RefObject } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { mapBlendshapesToEmotion } from "../emotionFromBlendshapes";
import type { EmotionTimelineSample, FacialTakeSummary } from "../types";
import { FACE_LANDMARKER_MODEL_URL, MEDIAPIPE_WASM_ROOT } from "../visionConstants";

/** Min seconds between timeline samples unless the label changes (keeps prompt size reasonable). */
const TIMELINE_MIN_GAP_SEC = 0.12;

function blendshapesFromResult(
  result: FaceLandmarkerResult,
): Record<string, number> | null {
  const first = result.faceBlendshapes?.[0];
  if (!first?.categories?.length) return null;
  const out: Record<string, number> = {};
  for (const c of first.categories) {
    out[c.categoryName] = c.score;
  }
  return out;
}

type Agg = {
  counts: Record<string, number>;
  framesSampled: number;
  framesWithFace: number;
  lastBlendshapes: Record<string, number> | null;
  timeline: EmotionTimelineSample[];
};

function emptyAgg(): Agg {
  return {
    counts: {},
    framesSampled: 0,
    framesWithFace: 0,
    lastBlendshapes: null,
    timeline: [],
  };
}

function summarizeAggregation(agg: Agg): FacialTakeSummary {
  if (agg.framesSampled === 0) {
    return {
      emotionCounts: {},
      framesSampled: 0,
      framesWithFace: 0,
      dominantEmotion: null,
      emotionTimeline: [],
      approximate: true,
    };
  }
  const dominant =
    Object.entries(agg.counts).sort((a, b) => b[1] - a[1])[0]?.[0] ?? null;
  return {
    emotionCounts: { ...agg.counts },
    framesSampled: agg.framesSampled,
    framesWithFace: agg.framesWithFace,
    dominantEmotion: dominant,
    emotionTimeline: agg.timeline.length ? [...agg.timeline] : undefined,
    snapshotBlendshapes: agg.lastBlendshapes ? { ...agg.lastBlendshapes } : undefined,
    approximate: agg.framesWithFace === 0,
  };
}

export function useFaceLandmarkerAnalysis(
  videoRef: RefObject<HTMLVideoElement | null>,
  mirror: boolean,
  isRecording: boolean,
) {
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const aggRef = useRef<Agg>(emptyAgg());
  const lastUiMs = useRef(0);
  const recordingT0Ref = useRef<number | null>(null);
  const lastTimelinePushRef = useRef<{ tSec: number; emotion: string } | null>(null);

  const [ready, setReady] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [liveEmotion, setLiveEmotion] = useState<string | null>(null);

  useEffect(() => {
    if (!isRecording) return;
    aggRef.current = emptyAgg();
    recordingT0Ref.current = performance.now();
    lastTimelinePushRef.current = null;
  }, [isRecording]);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      const opts = {
        baseOptions: {
          modelAssetPath: FACE_LANDMARKER_MODEL_URL,
          delegate: "GPU" as const,
        },
        runningMode: "VIDEO" as const,
        outputFaceBlendshapes: true,
        numFaces: 1,
      };
      try {
        const fileset = await FilesetResolver.forVisionTasks(MEDIAPIPE_WASM_ROOT);
        let landmarker: FaceLandmarker;
        try {
          landmarker = await FaceLandmarker.createFromOptions(fileset, opts);
        } catch {
          landmarker = await FaceLandmarker.createFromOptions(fileset, {
            ...opts,
            baseOptions: { ...opts.baseOptions, delegate: "CPU" },
          });
        }
        if (cancelled) {
          landmarker.close();
          return;
        }
        landmarkerRef.current = landmarker;
        setReady(true);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
      setReady(false);
    };
  }, []);

  useEffect(() => {
    if (!ready) return;
    let raf = 0;

    const loop = () => {
      const video = videoRef.current;
      const canvas = canvasRef.current;
      const landmarker = landmarkerRef.current;
      if (!video || !canvas || !landmarker) {
        raf = requestAnimationFrame(loop);
        return;
      }

      if (video.readyState < 2) {
        raf = requestAnimationFrame(loop);
        return;
      }

      const w = video.videoWidth;
      const h = video.videoHeight;
      if (w === 0 || h === 0) {
        raf = requestAnimationFrame(loop);
        return;
      }

      canvas.width = w;
      canvas.height = h;
      const ctx = canvas.getContext("2d");
      if (!ctx) {
        raf = requestAnimationFrame(loop);
        return;
      }

      ctx.save();
      if (mirror) {
        ctx.translate(w, 0);
        ctx.scale(-1, 1);
      }
      ctx.drawImage(video, 0, 0, w, h);
      ctx.restore();

      let result: FaceLandmarkerResult;
      try {
        result = landmarker.detectForVideo(canvas, performance.now());
      } catch {
        raf = requestAnimationFrame(loop);
        return;
      }

      const bs = blendshapesFromResult(result);
      let liveLabel: string | null = null;

      if (bs) {
        const { primaryEmotion } = mapBlendshapesToEmotion(bs);
        liveLabel = primaryEmotion;
        if (isRecording) {
          aggRef.current.framesSampled += 1;
          aggRef.current.framesWithFace += 1;
          aggRef.current.lastBlendshapes = bs;
          aggRef.current.counts[primaryEmotion] =
            (aggRef.current.counts[primaryEmotion] ?? 0) + 1;

          const t0 = recordingT0Ref.current;
          if (t0 != null) {
            const tSec = (performance.now() - t0) / 1000;
            const prev = lastTimelinePushRef.current;
            const gapOk = !prev || tSec - prev.tSec >= TIMELINE_MIN_GAP_SEC;
            const labelChanged = !prev || prev.emotion !== primaryEmotion;
            if (gapOk || labelChanged) {
              aggRef.current.timeline.push({ tSec, emotion: primaryEmotion });
              lastTimelinePushRef.current = { tSec, emotion: primaryEmotion };
            }
          }
        }
      } else if (isRecording) {
        aggRef.current.framesSampled += 1;
      }

      const now = Date.now();
      if (now - lastUiMs.current > 100) {
        lastUiMs.current = now;
        setLiveEmotion(liveLabel);
      }

      raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [ready, mirror, isRecording, videoRef]);

  const getSummary = (): FacialTakeSummary => summarizeAggregation(aggRef.current);

  return { canvasRef, ready, error, liveEmotion, getSummary };
}
