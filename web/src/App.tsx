import { useEffect, useRef, useState } from "react";
import { Disclaimer } from "./components/Disclaimer";
import { FeedbackSection } from "./components/FeedbackSection";
import { RecorderPanel } from "./components/RecorderPanel";
import { ScriptPanel } from "./components/ScriptPanel";
import { useFaceLandmarkerAnalysis } from "./hooks/useFaceLandmarkerAnalysis";
import { metricsFromRecording } from "./lib/audioMetrics";
import { generateRehearsalFeedbackFromBlob } from "./lib/rehearsalFeedback";
import { DEFAULT_GEMINI_MODEL } from "./lib/geminiModels";
import { pickAudioRecorderMime } from "./lib/recordingMime";
import type {
  FacialTakeSummary,
  RehearsalFeedbackResult,
  SuggestionRating,
} from "./types";
import "./App.css";

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null);
  const streamRef = useRef<MediaStream | null>(null);
  const recorderRef = useRef<MediaRecorder | null>(null);
  const chunksRef = useRef<Blob[]>([]);

  const scriptRef = useRef("");
  const intendedRef = useRef("");
  const guidanceRef = useRef(50);

  const [script, setScript] = useState("");
  const [intendedDirection, setIntendedDirection] = useState("");
  const [mirror, setMirror] = useState(true);
  const [streamReady, setStreamReady] = useState(false);
  const [mediaError, setMediaError] = useState<string | null>(null);

  const [theme, setTheme] = useState<"dark" | "light">(() => {
    return (localStorage.getItem("cl-theme") as "dark" | "light") ?? "dark";
  });

  const [isRecording, setIsRecording] = useState(false);
  const [recordingMs, setRecordingMs] = useState(0);

  const [guidanceSlider, setGuidanceSlider] = useState(50);
  const [analyzing, setAnalyzing] = useState(false);
  const [feedbackError, setFeedbackError] = useState<string | null>(null);
  const [feedback, setFeedback] = useState<RehearsalFeedbackResult | null>(null);
  const [ratings, setRatings] = useState<Record<string, SuggestionRating>>({});

  const {
    canvasRef: processCanvasRef,
    ready: faceModelReady,
    error: faceModelError,
    liveEmotion,
    getSummary: getFacialSummary,
  } = useFaceLandmarkerAnalysis(videoRef, mirror, isRecording);

  const apiKey = import.meta.env.VITE_GEMINI_API_KEY ?? "";
  const modelName = import.meta.env.VITE_GEMINI_MODEL ?? DEFAULT_GEMINI_MODEL;

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    localStorage.setItem("cl-theme", theme);
  }, [theme]);

  useEffect(() => {
    scriptRef.current = script;
    intendedRef.current = intendedDirection;
    guidanceRef.current = guidanceSlider;
  }, [script, intendedDirection, guidanceSlider]);

  useEffect(() => {
    let stream: MediaStream | null = null;
    (async () => {
      try {
        stream = await navigator.mediaDevices.getUserMedia({
          video: { facingMode: "user" },
          audio: true,
        });
        streamRef.current = stream;
        setStreamReady(true);
        setMediaError(null);
        const v = videoRef.current;
        if (v) {
          v.srcObject = stream;
          await v.play();
        }
      } catch (e) {
        setMediaError(e instanceof Error ? e.message : String(e));
      }
    })();
    return () => {
      stream?.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    };
  }, []);

  useEffect(() => {
    if (!isRecording) return;
    const t0 = Date.now();
    const id = window.setInterval(() => {
      setRecordingMs(Date.now() - t0);
    }, 100);
    return () => clearInterval(id);
  }, [isRecording]);

  const processRecording = async (
    blob: Blob,
    facialSummary: FacialTakeSummary | null,
  ) => {
    if (!import.meta.env.VITE_GEMINI_API_KEY) {
      setFeedbackError("Add VITE_GEMINI_API_KEY to web/.env to analyze takes.");
      return;
    }
    setAnalyzing(true);
    setFeedbackError(null);
    try {
      const metrics = await metricsFromRecording(blob);
      const result = await generateRehearsalFeedbackFromBlob(blob, {
        apiKey,
        modelName,
        script: scriptRef.current,
        intendedDirection: intendedRef.current,
        metrics,
        facialSummary,
        guidanceSlider: guidanceRef.current,
      });
      setFeedback(result);
      setRatings({});
    } catch (e) {
      setFeedbackError(e instanceof Error ? e.message : String(e));
    } finally {
      setAnalyzing(false);
    }
  };

  const startRecording = () => {
    const stream = streamRef.current;
    if (!stream) return;
    const audioTracks = stream.getAudioTracks();
    if (audioTracks.length === 0) {
      setFeedbackError("No microphone track found.");
      return;
    }
    chunksRef.current = [];
    const mime = pickAudioRecorderMime();
    const audioStream = new MediaStream(audioTracks);
    const mr = new MediaRecorder(audioStream, mime ? { mimeType: mime } : undefined);
    recorderRef.current = mr;
    mr.ondataavailable = (e) => {
      if (e.data.size > 0) chunksRef.current.push(e.data);
    };
    mr.onstop = () => {
      const blob = new Blob(chunksRef.current, { type: mr.mimeType });
      const facialSummary = getFacialSummary();
      void processRecording(blob, facialSummary);
    };
    setRecordingMs(0);
    setFeedback(null);
    setFeedbackError(null);
    mr.start();
    setIsRecording(true);
  };

  const stopRecording = () => {
    const mr = recorderRef.current;
    if (mr && mr.state !== "inactive") {
      mr.stop();
    }
    setIsRecording(false);
    recorderRef.current = null;
  };

  const onRate = (id: string, rating: Exclude<SuggestionRating, null>) => {
    setRatings((prev) => ({
      ...prev,
      [id]: prev[id] === rating ? null : rating,
    }));
  };

  return (
    <div className="app-root">
      <header className="hero">
        <div>
          <h1 className="main-title">CharacterLab</h1>
          <p className="sub-title">Solo rehearsal companion · voice-first</p>
        </div>
        <button
          type="button"
          className="theme-toggle"
          onClick={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        >
          {theme === "dark" ? "Light mode" : "Dark mode"}
        </button>
      </header>

      <Disclaimer />

      {!apiKey ? (
        <div className="banner warn">
          Set <code className="inline-code">VITE_GEMINI_API_KEY</code> in{" "}
          <code className="inline-code">web/.env</code> to generate partner feedback.
        </div>
      ) : null}

      {mediaError ? (
        <div className="banner error" role="alert">
          Could not access camera or microphone: {mediaError}
        </div>
      ) : null}

      {faceModelError ? (
        <div className="banner warn" role="status">
          Face ML unavailable ({faceModelError}). You can still record; feedback will use audio only.
        </div>
      ) : null}

      <div className="layout-main">
        <div className="col-left">
          <ScriptPanel
            script={script}
            intendedDirection={intendedDirection}
            onScriptChange={setScript}
            onIntendedChange={setIntendedDirection}
          />
        </div>
        <div className="col-center">
          <RecorderPanel
            videoRef={videoRef}
            processCanvasRef={processCanvasRef}
            mirror={mirror}
            onMirrorChange={setMirror}
            isRecording={isRecording}
            recordingMs={recordingMs}
            onStart={startRecording}
            onStop={stopRecording}
            canRecord={streamReady && !analyzing}
            liveFaceLabel={liveEmotion}
            faceModelReady={faceModelReady}
            faceModelFailed={faceModelError !== null}
          />
        </div>
        <div className="col-right">
          <FeedbackSection
            guidanceSlider={guidanceSlider}
            onGuidanceChange={setGuidanceSlider}
            analyzing={analyzing}
            error={feedbackError}
            feedback={feedback}
            ratings={ratings}
            onRate={onRate}
          />
        </div>
      </div>
    </div>
  );
}
