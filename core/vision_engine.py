import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import numpy as np
import time
from .schema import VisionEmotion

class VisionEngine:
    def __init__(self, model_path="models/face_landmarker.task"):
        """
        Initializes the MediaPipe Face Landmarker for 52 Blendshape tracking.
        Optimized for CPU processing on macOS.
        """
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_face_transformation_matrixes=True,
            running_mode=vision.RunningMode.VIDEO)
        self.detector = vision.FaceLandmarker.create_from_options(options)
        self.fps_throttle = 1  # Logic can be adjusted for performance
        self.frame_count = 0

    def process_frame(self, frame_rgb: np.ndarray, timestamp_ms: int) -> VisionEmotion:
        """
        Processes a single video frame. 
        frame_rgb should be an RGB numpy array.
        """
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        
        # Inference
        result = self.detector.detect_for_video(mp_image, timestamp_ms)
        
        if not result.face_blendshapes:
            return None

        # Extract 52 blendshapes for the first face detected
        blendshapes_list = result.face_blendshapes[0]
        blendshapes_dict = {b.category_name: b.score for b in blendshapes_list}
        
        emotion, confidence = self._map_to_emotion(blendshapes_dict)
        
        return VisionEmotion(
            timestamp=time.time(),
            blendshapes=blendshapes_dict,
            primary_emotion=emotion,
            confidence=float(confidence)
        )

    def _map_to_emotion(self, bs: dict):
        """
        Heuristic mapping of MediaPipe Blendshapes to discrete emotions.
        """
        scores = {
            "Joy": (bs.get("mouthSmileLeft", 0) + bs.get("mouthSmileRight", 0)) / 2,
            "Sadness": (bs.get("browInnerUp", 0) + bs.get("mouthFrownLeft", 0) + bs.get("mouthFrownRight", 0)) / 3,
            "Anger": (bs.get("browDownLeft", 0) + bs.get("browDownRight", 0) + bs.get("mouthPressLeft", 0)) / 3,
            "Surprise": (bs.get("browOuterUpLeft", 0) + bs.get("browOuterUpRight", 0) + bs.get("jawOpen", 0)) / 3,
            "Disgust": (bs.get("noseSneerLeft", 0) + bs.get("noseSneerRight", 0)) / 2,
            "Fear": (bs.get("browInnerUp", 0) + bs.get("eyeWideLeft", 0) + bs.get("eyeWideRight", 0)) / 3,
        }
        
        primary = max(scores, key=scores.get)
        confidence = scores[primary]
        
        if confidence < 0.15:
            return "Neutral", 1.0
            
        return primary, confidence
