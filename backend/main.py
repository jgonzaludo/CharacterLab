"""
Local FFEM-compatible facial emotion analysis for CharacterLab.

Uses DeepFace.analyze(..., actions=['emotion']) with MediaPipe face detection,
matching the approach in Fast Facial Emotion Monitoring (FFEM):
https://github.com/WiseGeorge/Fast-Facial-Emotion-Monitoring-FFEM-Package

Not suitable for production deployment without auth, size limits, and hardening.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import cv2
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="CharacterLab FFEM bridge")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lazy import so `uvicorn main:app` can load before heavy ML deps finish installing
_deepface = None


def get_deepface():
    global _deepface
    if _deepface is None:
        from deepface import DeepFace as DF

        _deepface = DF
    return _deepface


SAMPLE_EVERY_SEC = 0.18
DETECTOR_BACKEND = os.environ.get("FFEM_DETECTOR_BACKEND", "mediapipe")


def analyze_video_frames(path: str) -> dict:
    """
    Sample frames from a video file and run DeepFace emotion analysis (FFEM-style).
    Returns a JSON-serializable structure aligned with the web app's FacialTakeSummary.
    """
    DeepFace = get_deepface()
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Could not open video file")

    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    step_frames = max(1, int(round(fps * SAMPLE_EVERY_SEC)))

    timeline: list[dict] = []
    counts: dict[str, int] = {}
    frames_sampled = 0
    frames_with_face = 0

    frame_i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_i % step_frames == 0:
            frames_sampled += 1
            t_sec = frame_i / fps
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            try:
                result = DeepFace.analyze(
                    rgb,
                    actions=["emotion"],
                    enforce_detection=False,
                    detector_backend=DETECTOR_BACKEND,
                )
                row = result[0] if isinstance(result, list) else result
                dom = str(row.get("dominant_emotion", "neutral")).lower()
                timeline.append({"tSec": round(float(t_sec), 2), "emotion": dom})
                counts[dom] = counts.get(dom, 0) + 1
                frames_with_face += 1
            except Exception:
                pass

        frame_i += 1

    cap.release()

    dominant = None
    if counts:
        dominant = max(counts, key=counts.get)

    return {
        "emotionCounts": counts,
        "framesSampled": frames_sampled,
        "framesWithFace": frames_with_face,
        "dominantEmotion": dominant,
        "emotionTimeline": timeline,
        "approximate": frames_with_face == 0,
        "engine": "deepface-ffem-style",
    }


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    suffix = Path(file.filename or "take.webm").suffix or ".webm"
    raw = await file.read()
    if len(raw) > 80 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="File too large (max 80MB)")

    fd, path = tempfile.mkstemp(suffix=suffix)
    try:
        os.write(fd, raw)
        os.close(fd)
        return analyze_video_frames(path)
    finally:
        try:
            os.unlink(path)
        except OSError:
            pass
