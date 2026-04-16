import torch
import numpy as np
import time
import os
import librosa
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from transformers import pipeline
from .schema import AudioEmotion

load_dotenv()

class AudioEngine:
    AUDIO_TONE_MAP = {
        "happy": "Warm / positive",
        "sad": "Quiet / reflective",
        "angry": "Harsh / tense",
        "fear": "Anxious / uncertain",
        "neutral": "Balanced / calm",
        "excited": "Energetic / lively",
        "surprise": "Surprised / curious",
        "disgust": "Dismissive / negative"
    }

    QUESTION_WORDS = {"who", "what", "when", "where", "why", "how", "is", "are", "do", "does", "did", "can", "could", "would", "should"}

    def __init__(self, whisper_model_size="tiny", emotion_model_id="harshit345/xlsr-wav2vec-speech-emotion-recognition"):
        """
        Initializes Faster-Whisper and Transformers pipeline for audio analysis.
        """
        self.whisper = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")

        hf_token = os.getenv("HUGGINGFACE_API_TOKEN") or os.getenv("HF_TOKEN")
        self.classifier = pipeline("audio-classification", model=emotion_model_id, token=hf_token)
        self.text_analyzer = pipeline("sentiment-analysis", return_all_scores=True)

        self.target_sr = 16000
        self.buffer = np.array([], dtype=np.float32)

    def _extract_keywords(self, text: str) -> list[str]:
        stop_words = {
            "that", "this", "there", "their", "about", "would", "could", "should",
            "have", "with", "your", "from", "which", "what", "when", "where",
            "they", "them", "then", "been", "were", "will", "just", "like"
        }
        words = [w.strip(".,?!;:\"'()[]") for w in text.lower().split()]
        keywords = [w for w in words if len(w) > 3 and w not in stop_words]
        return list(dict.fromkeys(keywords))[:6]

    def _detect_intent(self, text: str) -> str:
        normalized = text.strip().lower()
        if not normalized:
            return "Unknown"

        if normalized.endswith("?") or any(w in normalized.split()[:3] for w in self.QUESTION_WORDS):
            return "Question"
        if normalized.startswith(("please", "could you", "would you", "can you", "i need", "i want")):
            return "Request"
        if any(token in normalized for token in ("need", "want", "must", "should", "have to")):
            return "Need"
        return "Statement"

    def _combine_tone(self, audio_label: str, sentiment_label: str) -> str:
        lower_audio = audio_label.lower()
        audio_tone = self.AUDIO_TONE_MAP.get(lower_audio, lower_audio.title())

        if sentiment_label.upper() == "NEGATIVE":
            return f"{audio_tone} with negative tone"
        if sentiment_label.upper() == "POSITIVE":
            return f"{audio_tone} with positive tone"
        return audio_tone

    def process_audio(self, audio_chunk: np.ndarray, source_sr: int = 16000) -> AudioEmotion:
        """
        Processes a chunk of audio. Resamples to 16kHz if necessary.
        Runs inference when identifying ~2.5 seconds of audio.
        """
        if source_sr != self.target_sr:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=source_sr, target_sr=self.target_sr)

        self.buffer = np.append(self.buffer, audio_chunk)

        if len(self.buffer) < self.target_sr * 2.5:
            return None

        segment = self.buffer.copy()
        self.buffer = self.buffer[-int(self.target_sr * 0.5):]

        segments, _ = self.whisper.transcribe(segment, beam_size=2, language="en")
        transcription = " ".join([s.text for s in segments]).strip()
        if not transcription:
            return None

        results = self.classifier(segment)
        if not results:
            return None

        primary_emotion = results[0]["label"]
        confidence = float(results[0]["score"])
        prosody = {r["label"]: float(r["score"]) for r in results}

        sentiment_results = self.text_analyzer(transcription)
        sentiment_scores = {r["label"]: float(r["score"]) for r in sentiment_results[0]}
        sentiment_label = sentiment_results[0][0]["label"] if sentiment_results and sentiment_results[0] else "neutral"

        tone = self._combine_tone(primary_emotion, sentiment_label)
        intent = self._detect_intent(transcription)
        keywords = self._extract_keywords(transcription)

        return AudioEmotion(
            timestamp=time.time(),
            prosody=prosody,
            transcription=transcription,
            sentiment=sentiment_label,
            sentiment_scores=sentiment_scores,
            tone=tone,
            intent=intent,
            keywords=keywords,
            primary_emotion=primary_emotion,
            confidence=confidence,
        )
