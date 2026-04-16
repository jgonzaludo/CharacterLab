# 🎭 CharacterLab: Real-Time Emotion Mirror

CharacterLab is a multimodal rehearsal assistant designed to help actors and public speakers analyze their facial expressions and vocal prosody in real-time.

---

## 🚀 Quick Start

### 1. Prerequisites
- **Python 3.9+**
- A **Hugging Face Read Token** (for downloading the audio emotion model)
- A **Google Gemini API Key** (for Phase 2 coaching feedback)

### 2. Installation
Clone the repository and install the dependencies:
```bash
pip install -r requirements.txt
```

### 3. Configuration
Create a `.env` file in the root directory (or update the existing one) with your credentials:
```env
HF_TOKEN=your_hugging_face_read_token
GEMINI_API_KEY=your_google_gemini_api_key
```

### 4. Running the App
Launch the Streamlit interface:
```bash
streamlit run app.py
```

---

## 🛠 Features (Phase 1)
- **Real-Time Facial Analysis**: tracks 52 MediaPipe blendshapes to predict primary emotions (Joy, Sadness, Anger, Surprise, etc.).
- **Vocal Prosody Tracking**: Uses `Wav2Vec2` for speech emotion recognition and `Faster-Whisper` for live transcription.
- **Mirror Mode**: Flip your camera directly in the UI for a natural rehearsal experience.
- **Multimodal Sync**: Zero-lag synchronization between camera overlays and statistical metrics.

---

## 📂 Project Structure
- `app.py`: Main Streamlit application and UI logic.
- `core/`:
  - `vision_engine.py`: MediaPipe Face Landmarker implementation.
  - `audio_engine.py`: Faster-Whisper and Emotion Classification logic.
  - `schema.py`: Data models for cross-engine communication.
- `models/`: Directory for local model weights (e.g., `face_landmarker.task`).
