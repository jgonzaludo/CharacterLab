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
    sentiment: Optional[str] = Field(None, description="Text sentiment label derived from speech transcription")
    sentiment_scores: Optional[Dict[str, float]] = Field(None, description="Confidence scores for text sentiment analysis")
    tone: Optional[str] = Field(None, description="Combined tone summary from audio expressiveness and text sentiment")
    intent: Optional[str] = Field(None, description="Detected communicative intent such as question, request, or statement")
    keywords: Optional[List[str]] = Field(None, description="Extracted keywords from the transcription")
    primary_emotion: str = Field(..., description="Detected vocal emotion")
    confidence: float = Field(..., description="Confidence score for the detected vocal emotion")

class EmotionState(BaseModel):
    timestamp: float = Field(default_factory=lambda: datetime.now().timestamp())
    vision: Optional[VisionEmotion] = None
    audio: Optional[AudioEmotion] = None
