# FFEM bridge (Python)

This service implements the same **DeepFace `analyze(..., actions=['emotion'])`** pattern as [Fast Facial Emotion Monitoring (FFEM)](https://github.com/WiseGeorge/Fast-Facial-Emotion-Monitoring-FFEM-Package) so the web app can attach timed emotion cues to the Gemini prompt.

## Setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

First run may download DeepFace / detector weights.

## Run

```bash
uvicorn main:app --reload --port 8787
```

Point the web app at `VITE_FFEM_URL=http://127.0.0.1:8787` in `web/.env`.

## Notes

- OpenCV must be able to decode the browser’s `video/webm` recording (depends on your OpenCV build).
- Override face detector with env `FFEM_DETECTOR_BACKEND` (default `mediapipe`; try `opencv` if needed).
- This endpoint is for **local class demos** only: no authentication, coarse file-size limit.
