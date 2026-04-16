from pydantic import BaseModel, Field
from typing import Dict, List, Optional
from datetime import datetime

class VisionEmotion(BaseModel):
    timestamp: float = Field(..., description="Unix timestamp of the frame")
    blendshapes: Dict[str, float] = Field(..., description="Raw blendshape coefficients from MediaPipe")
    primary_emotion: str = Field(..., description="Detected primary emotion (e.g., Happy, Sad, Neutral)")
    confidence: float = Field(..., description="Confidence score for the detected emotion")

class AudioEmotion(BaseModel):
    timestamp: float = Field(..., description="Unix timestamp of the audio segment")
    prosody: Dict[str, float] = Field(..., description="Vocal prosody scores from SpeechBrain")
    transcription: str = Field(..., description="Transcribed text from Faster-Whisper")
    primary_emotion: str = Field(..., description="Detected vocal emotion")
    confidence: float = Field(..., description="Confidence score for the detected vocal emotion")

class EmotionState(BaseModel):
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    vision: Optional[VisionEmotion] = None
    audio: Optional[AudioEmotion] = None
