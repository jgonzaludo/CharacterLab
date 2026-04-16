import torch
import numpy as np
import time
import os
from dotenv import load_dotenv
from faster_whisper import WhisperModel
from transformers import pipeline
from .schema import AudioEmotion

load_dotenv()

class AudioEngine:
    def __init__(self, whisper_model_size="tiny", emotion_model_id="harshit345/xlsr-wav2vec-speech-emotion-recognition"):
        """
        Initializes Faster-Whisper and Transformers pipeline for audio analysis.
        Uses Wav2Vec2 for emotion detection to avoid proprietary SpeechBrain lazy import issues.
        """
        # Faster-Whisper on CPU
        self.whisper = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        
        # Audio Classification Pipeline (Wav2Vec2)
        # Load token from environment
        hf_token = os.getenv("HF_TOKEN")
        
        # This will download the model (~300MB) on first run
        self.classifier = pipeline("audio-classification", model=emotion_model_id, token=hf_token)
        
        self.sample_rate = 16000
        self.buffer = np.array([], dtype=np.float32)

    def process_audio(self, audio_chunk: np.ndarray) -> AudioEmotion:
        """
        Processes a chunk of audio (16kHz mono).
        Runs inference every 2 seconds.
        """
        self.buffer = np.append(self.buffer, audio_chunk)
        
        # Minimum 2 seconds for meaningful analysis
        if len(self.buffer) < self.sample_rate * 2:
            return None
            
        # Extract last 2 seconds
        segment = self.buffer[-self.sample_rate * 2:]
        # Overlap buffer
        self.buffer = self.buffer[-self.sample_rate:]
        
        # 1. Faster-Whisper Transcription
        segments, _ = self.whisper.transcribe(segment, beam_size=1)
        transcription = " ".join([s.text for s in segments]).strip()
        
        # 2. Emotion Recognition (Wav2Vec2)
        # pipeline accepts numpy array directly
        results = self.classifier(segment)
        
        if not results:
             return None
             
        primary_emotion = results[0]['label']
        confidence = float(results[0]['score'])
        prosody = {r['label']: float(r['score']) for r in results}

        return AudioEmotion(
            timestamp=time.time(),
            prosody=prosody,
            transcription=transcription,
            primary_emotion=primary_emotion,
            confidence=confidence
        )
