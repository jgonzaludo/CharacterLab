import torch
import numpy as np
import time
from faster_whisper import WhisperModel
from speechbrain.inference.interfaces import foreign_class
from .schema import AudioEmotion

class AudioEngine:
    def __init__(self, whisper_model_size="tiny", sb_model_source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP"):
        """
        Initializes Faster-Whisper and SpeechBrain for low-latency audio processing.
        """
        # Faster-Whisper on CPU
        self.whisper = WhisperModel(whisper_model_size, device="cpu", compute_type="int8")
        
        # SpeechBrain Emotion Recognition
        # This will download models to ~/.cache/huggingface
        self.classifier = foreign_class(
            source=sb_model_source, 
            pymodule_file="custom_interface.py", 
            classname="EncoderClassifier"
        )
        
        self.sample_rate = 16000
        self.buffer = np.array([], dtype=np.float32)

    def process_audio(self, audio_chunk: np.ndarray) -> AudioEmotion:
        """
        Processes a chunk of audio (nominally 16kHz mono).
        Accumulates buffer and runs inference every 2 seconds.
        """
        self.buffer = np.append(self.buffer, audio_chunk)
        
        # Minimum 2 seconds for meaningful emotion analysis
        if len(self.buffer) < self.sample_rate * 2:
            return None
            
        # Extract last 2 seconds
        segment = self.buffer[-self.sample_rate * 2:]
        # Retain last 1 second for overlap
        self.buffer = self.buffer[-self.sample_rate:]
        
        # 1. Faster-Whisper Transcription
        segments, _ = self.whisper.transcribe(segment, beam_size=1)
        transcription = " ".join([s.text for s in segments]).strip()
        
        # 2. SpeechBrain Emotion (Prosody)
        # Convert to tensor and add batch dim
        signal = torch.from_numpy(segment).unsqueeze(0)
        out_prob, score, index, text_lab = self.classifier.classify_batch(signal)
        
        primary_emotion = text_lab[0]
        confidence = float(torch.exp(out_prob).max())
        
        # Simple prosody breakdown (mapped from internal labels)
        prosody = {
            "neu": 0.0, "hap": 0.0, "sad": 0.0, "ang": 0.0
        }
        # IEMOCAP labels: neu, hap, sad, ang
        for i, label in enumerate(text_lab):
             if label in prosody:
                 prosody[label] = float(torch.exp(out_prob)[0][i])

        return AudioEmotion(
            timestamp=time.time(),
            prosody=prosody,
            transcription=transcription,
            primary_emotion=primary_emotion,
            confidence=confidence
        )
