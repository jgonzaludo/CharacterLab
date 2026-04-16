import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import time
import queue
import threading
from core.vision_engine import VisionEngine
from core.audio_engine import AudioEngine
from core.schema import EmotionState

# --- Page Config & Styling ---
st.set_page_config(
    page_title="CharacterLab | Emotion Mirror",
    page_icon="🎭",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    .reportview-container {
        background: #0f1117;
    }
    .main {
        background: #0f1117;
        color: #e0e0e0;
    }
    .stMetric {
        background: rgba(255, 255, 255, 0.05);
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #6366f1;
    }
    h1, h2, h3 {
        color: #ffffff !important;
    }
</style>
""", unsafe_allow_html=True)

st.title("🎭 CharacterLab: Real-Time Emotion Mirror")
st.caption("Phase 1: Multimodal Sensing Foundation")

# --- Shared State for Callbacks ---
class AppState:
    flip_camera = False
    vision_result = None
    audio_result = None

app_state = AppState()

# --- Engine Initialization ---
@st.cache_resource
def load_engines():
    return VisionEngine(), AudioEngine()

try:
    vision_engine, audio_engine = load_engines()
except Exception as e:
    st.error(f"Error loading engines: {e}")
    st.stop()

# --- Callbacks ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")
    
    # Flip if requested
    if app_state.flip_camera:
        img = cv2.flip(img, 1)

    # Process for vision (MediaPipe expects RGB)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(time.time() * 1000)
    
    try:
        vision_state = vision_engine.process_frame(rgb_img, timestamp_ms)
        if vision_state:
            app_state.vision_result = vision_state
            
            # Simple Mesh/Text Overlay
            color = (0, 255, 0) if vision_state.primary_emotion != "Neutral" else (200, 200, 200)
            cv2.putText(img, f"EMOTION: {vision_state.primary_emotion}", (30, 50), 
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, color, 2)
            cv2.putText(img, f"CONF: {vision_state.confidence:.1%}", (30, 90), 
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, color, 1)
    except Exception as e:
        print(f"Vision error: {e}")

    return av.VideoFrame.from_ndarray(img, format="bgr24")

def audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
    try:
        audio_data = frame.to_ndarray().flatten().astype(np.float32) / 32768.0
        # Pass the actual sample rate to handle resampling
        audio_state = audio_engine.process_audio(audio_data, source_sr=frame.sample_rate)
        if audio_state:
            app_state.audio_result = audio_state
    except Exception as e:
        print(f"Audio error: {e}")
        
    return frame

# --- UI Layout ---
col_vid, col_stats = st.columns([2, 1])

with col_vid:
    app_state.flip_camera = st.checkbox("Flip Camera (Mirror Mode)", value=True)
    webrtc_ctx = webrtc_streamer(
        key="emotion-mirror",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        audio_frame_callback=audio_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": True},
        async_processing=True,
    )

with col_stats:
    st.subheader("Live Analysis")
    
    v_metric = st.empty()
    a_metric = st.empty()
    transcription_box = st.empty()
    
    with st.expander("Raw Blendshapes", expanded=False):
        bs_chart = st.empty()

# --- Main UI Update Loop ---
if webrtc_ctx.state.playing:
    while True:
        # Update Vision Metrics
        if app_state.vision_result:
            vs = app_state.vision_result
            v_metric.metric("Facial Vibe", vs.primary_emotion, f"{vs.confidence:.1%} match")
            
            # Show top 5 blendshapes
            top_bs = dict(sorted(vs.blendshapes.items(), key=lambda x: x[1], reverse=True)[:5])
            bs_chart.write(top_bs)

        # Update Audio Metrics
        if app_state.audio_result:
            as_ = app_state.audio_result
            a_metric.metric("Vocal Vibe", as_.primary_emotion, f"{as_.confidence:.1%} match")
            transcription_box.success(f"**Transcript:** {as_.transcription}")

        time.sleep(0.1)
