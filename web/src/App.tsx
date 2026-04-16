import { useEffect, useRef, useState } from "react";
import {
  FaceLandmarker,
  FilesetResolver,
  type FaceLandmarkerResult,
} from "@mediapipe/tasks-vision";
import { EMOTION_COLOR, EMOTION_EMOJI, FACE_LANDMARKER_MODEL_URL, MEDIAPIPE_WASM_ROOT } from "./constants";
import { mapBlendshapesToEmotion } from "./emotionFromBlendshapes";
import type { VisionEmotion } from "./types";
import "./App.css";

function classificationsToBlendshapes(result: FaceLandmarkerResult): Record<string, number> | null {
  const first = result.faceBlendshapes?.[0];
  if (!first?.categories?.length) return null;
  const out: Record<string, number> = {};
  for (const c of first.categories) {
    out[c.categoryName] = c.score;
  }
  return out;
}

function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const landmarkerRef = useRef<FaceLandmarker | null>(null);
  const lastUiMsRef = useRef(0);

  const [mirror, setMirror] = useState(true);
  const [vision, setVision] = useState<VisionEmotion | null>(null);
  const [loadError, setLoadError] = useState<string | null>(null);
  const [cameraError, setCameraError] = useState<string | null>(null);
  const [engineReady, setEngineReady] = useState(false);

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
        setEngineReady(true);
      } catch (e) {
        setLoadError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      cancelled = true;
      landmarkerRef.current?.close();
      landmarkerRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!engineReady) return;
    const video = videoRef.current;
    if (!video) return;

    let cancelled = false;
    (async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: false,
        });
        if (cancelled) {
          stream.getTracks().forEach((t) => t.stop());
          return;
        }
        video.srcObject = stream;
        await video.play();
        setCameraError(null);
      } catch (e) {
        setCameraError(e instanceof Error ? e.message : String(e));
      }
    })();

    return () => {
      cancelled = true;
      const stream = video.srcObject as MediaStream | null;
      stream?.getTracks().forEach((t) => t.stop());
      video.srcObject = null;
    };
  }, [engineReady]);

  useEffect(() => {
    if (!engineReady) return;
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

      const ts = performance.now();
      let result: FaceLandmarkerResult;
      try {
        result = landmarker.detectForVideo(canvas, ts);
      } catch {
        raf = requestAnimationFrame(loop);
        return;
      }

      const blendshapes = classificationsToBlendshapes(result);
      let overlayVision: VisionEmotion | null = null;

      if (blendshapes) {
        const { primaryEmotion, confidence } = mapBlendshapesToEmotion(blendshapes);
        overlayVision = {
          timestamp: Date.now() / 1000,
          blendshapes,
          primaryEmotion,
          confidence,
        };

        const color = EMOTION_COLOR[primaryEmotion as keyof typeof EMOTION_COLOR] ?? "#6366f1";
        const emoji = EMOTION_EMOJI[primaryEmotion as keyof typeof EMOTION_EMOJI] ?? "";
        const pct = Math.round(confidence * 100);
        const label = `${emoji} ${primaryEmotion}  ${pct}%`;

        ctx.save();
        ctx.setTransform(1, 0, 0, 1, 0, 0);
        ctx.font = "600 20px Syne, system-ui, sans-serif";
        ctx.fillStyle = color;
        ctx.shadowColor = "rgba(0,0,0,0.85)";
        ctx.shadowBlur = 6;
        ctx.fillText(label, 20, 40);
        ctx.restore();
      }

      const now = Date.now();
      if (now - lastUiMsRef.current >= 100) {
        lastUiMsRef.current = now;
        setVision(overlayVision);
      }

      raf = requestAnimationFrame(loop);
    };

    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [engineReady, mirror]);

  return (
    <div className="app-root">
      {loadError && (
        <div className="banner error">
          Failed to load vision engine: {loadError}
        </div>
      )}
      {cameraError && !loadError && (
        <div className="banner error">Camera: {cameraError}</div>
      )}

      <header className="hero">
        <h1 className="main-title">CharacterLab</h1>
        <p className="sub-title">Real-Time Emotion Mirror · Vision Engine</p>
      </header>

      <div className="layout">
        <section className="video-column">
          <label className="mirror-toggle">
            <input
              type="checkbox"
              checked={mirror}
              onChange={(e) => setMirror(e.target.checked)}
            />
            Mirror mode
          </label>
          <div className="video-shell">
            <video ref={videoRef} className="hidden-video" playsInline muted />
            <canvas ref={canvasRef} className="mirror-canvas" />
            {!engineReady && !loadError && (
              <p className="overlay-hint">Loading vision model…</p>
            )}
          </div>
        </section>

        <aside className="stats-column">
          <div className="section-label">Face</div>
          <div
            className="emotion-card"
            style={
              vision
                ? {
                    borderColor: `${EMOTION_COLOR[vision.primaryEmotion as keyof typeof EMOTION_COLOR] ?? "#6366f1"}33`,
                  }
                : undefined
            }
          >
            <div className="emotion-label">Detected Emotion</div>
            {vision ? (
              <>
                <div
                  className="emotion-value"
                  style={{
                    color:
                      EMOTION_COLOR[vision.primaryEmotion as keyof typeof EMOTION_COLOR] ??
                      "#f8fafc",
                  }}
                >
                  {EMOTION_EMOJI[vision.primaryEmotion as keyof typeof EMOTION_EMOJI]}{" "}
                  {vision.primaryEmotion}
                </div>
                <div className="emotion-conf">
                  {(vision.confidence * 100).toFixed(1)}% confidence
                </div>
              </>
            ) : (
              <>
                <div className="emotion-value muted">—</div>
                <div className="emotion-conf">Waiting for camera…</div>
              </>
            )}
          </div>
        </aside>
      </div>
    </div>
  );
}

export default App;
