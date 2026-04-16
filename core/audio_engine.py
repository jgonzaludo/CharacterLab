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
    def __init__(self, whisper_model_size="tiny", emotion_model_id="harshit345/xlsr-wav2vec-speech-emotion-recognition"):
        """
        Initializes Faster-Whisper and Transformers pipeline for audio analysis.
        """
        # Faster-Whisper on CPU
        self.whisper = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        
        # Audio Classification Pipeline (Wav2Vec2)
        hf_token = os.getenv("HF_TOKEN")
        self.classifier = pipeline("audio-classification", model=emotion_model_id, token=hf_token)
        
        self.target_sr = 16000
        self.buffer = np.array([], dtype=np.float32)

    def process_audio(self, audio_chunk: np.ndarray, source_sr: int = 16000) -> AudioEmotion:
        """
        Processes a chunk of audio. Resamples to 16kHz if necessary.
        Runs inference when identifying ~2.5 seconds of audio.
        """
        # Resample if source SR isn't 16k
        if source_sr != self.target_sr:
            audio_chunk = librosa.resample(audio_chunk, orig_sr=source_sr, target_sr=self.target_sr)

        self.buffer = np.append(self.buffer, audio_chunk)
        
        # We need at least 2.5 seconds for good whisper/emotion context
        if len(self.buffer) < self.target_sr * 2.5:
            return None
            
        # Extract the segment to process
        segment = self.buffer.copy()
        # Keep a 0.5s overlap for the next window to avoid cutting words
        self.buffer = self.buffer[-int(self.target_sr * 0.5):]
        
        # 1. Faster-Whisper Transcription
        # We use a slightly larger beam_size for better accuracy
        segments, _ = self.whisper.transcribe(segment, beam_size=2, language="en")
        transcription = " ".join([s.text for s in segments]).strip()
        
        if not transcription:
            return None

        # 2. Emotion Recognition (Wav2Vec2)
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
