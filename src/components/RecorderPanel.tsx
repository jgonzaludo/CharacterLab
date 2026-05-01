import type { RefObject } from "react";

type Props = {
  videoRef: RefObject<HTMLVideoElement | null>;
  processCanvasRef: RefObject<HTMLCanvasElement | null>;
  mirror: boolean;
  onMirrorChange: (v: boolean) => void;
  isRecording: boolean;
  recordingMs: number;
  onStart: () => void;
  onStop: () => void;
  onCancel: () => void;
  canRecord: boolean;
  liveFaceLabel: string | null;
  faceModelReady: boolean;
  faceModelFailed: boolean;
};

function formatMs(ms: number): string {
  const s = Math.floor(ms / 1000);
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export function RecorderPanel({
  videoRef,
  processCanvasRef,
  mirror,
  onMirrorChange,
  isRecording,
  recordingMs,
  onStart,
  onStop,
  onCancel,
  canRecord,
  liveFaceLabel,
  faceModelReady,
  faceModelFailed,
}: Props) {
  return (
    <section className="panel recorder-panel">
      <div className="recorder-toolbar">
        <label className="mirror-toggle">
          <input
            type="checkbox"
            checked={mirror}
            onChange={(e) => onMirrorChange(e.target.checked)}
          />
          Mirror camera
        </label>
        <div className="recorder-status">
          {isRecording ? (
            <span className="rec-dot" aria-hidden />
          ) : null}
          <span className="timer">{formatMs(recordingMs)}</span>
        </div>
      </div>

      <div className={`video-shell ${mirror ? "mirror" : ""}`}>
        <video ref={videoRef} className="preview-video" playsInline muted />
        <canvas ref={processCanvasRef} className="process-canvas" aria-hidden />
        {!canRecord ? (
          <p className="overlay-hint">Waiting for camera and microphone…</p>
        ) : !faceModelReady && !faceModelFailed ? (
          <p className="overlay-hint overlay-hint-subtle">Loading face model…</p>
        ) : null}
        {canRecord && (faceModelReady || faceModelFailed) ? (
          <div className="face-pill" title="Heuristic ML label from blendshapes (not clinical)">
            {faceModelFailed ? "Face (ML): off" : `Face (ML): ${liveFaceLabel ?? "—"}`}
          </div>
        ) : null}
      </div>

      <div className="recorder-actions">
        {!isRecording ? (
          <button
            type="button"
            className="btn btn-primary"
            disabled={!canRecord}
            onClick={onStart}
          >
            Record take
          </button>
        ) : (
          <div className="recording-buttons">
            <button type="button" className="btn btn-stop" onClick={onStop}>
              Stop
            </button>
            <button type="button" className="btn btn-cancel" onClick={onCancel}>
              Cancel take
            </button>
          </div>
        )}
        <p className="recorder-hint">
          Audio drives prosody analysis. MediaPipe face blendshapes are summarized over your take
          and sent to the partner model as soft context (not a score).
        </p>
      </div>
    </section>
  );
}
