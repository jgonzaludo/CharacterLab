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

# Custom Theatrical Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&display=swap');

    :root {
        --bg-deep: #050505;
        --accent-glow: #fbbf24; /* Amber Spotlight */
        --card-bg: rgba(255, 255, 255, 0.03);
        --text-main: #f8fafc;
        --text-dim: #94a3b8;
    }

    .stApp {
        background: radial-gradient(circle at 20% 20%, rgba(251, 191, 36, 0.05) 0%, transparent 40%),
                    radial-gradient(circle at 80% 80%, rgba(99, 102, 241, 0.05) 0%, transparent 40%),
                    var(--bg-deep);
        color: var(--text-main);
        font-family: 'Syne', sans-serif !important;
    }

    /* Target specific text elements to avoid breaking icons/expanders */
    .stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp label, .main-title, .sub-title, .emotion-card {
        font-family: 'Syne', sans-serif !important;
    }

    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        letter-spacing: -2px;
        background: linear-gradient(to right, #ffffff, #94a3b8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0rem;
        animation: fadeInDown 0.8s ease-out;
    }

    .sub-title {
        color: var(--accent-glow);
        text-transform: uppercase;
        letter-spacing: 4px;
        font-size: 0.8rem;
        font-weight: 700;
        margin-bottom: 2rem;
        animation: fadeInDown 1s ease-out;
    }

    /* Glassmorphism Cards */
    .emotion-card {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out both;
    }
    
    .emotion-card:hover {
        border-color: var(--accent-glow);
        transform: translateY(-5px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .metric-label {
        color: var(--text-dim);
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: var(--text-main);
        font-size: 2rem;
        font-weight: 800;
        margin-top: 5px;
    }

    .metric-delta {
        color: var(--accent-glow);
        font-size: 0.85rem;
        font-weight: 600;
    }

    /* Video Container Styling */
    .element-container iframe, .stWebRtcStreamer {
        border-radius: 24px;
        border: 2px solid rgba(255,255,255,0.05);
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.8);
    }

    /* Animation Keyframes */
    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    /* Hide generic Streamlit elements */
    #MainMenu, footer, header {visibility: hidden;}
    .stDeployButton {display:none;}
    
    /* Staggered load effect */
    .st-emotion-1 { animation-delay: 0.1s; }
    .st-emotion-2 { animation-delay: 0.2s; }
    .st-emotion-3 { animation-delay: 0.3s; }
</style>

<div class="main-title">CharacterLab</div>
<div class="sub-title">Phase 1: Deep Sensing Neural Mirror</div>
""", unsafe_allow_html=True)

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
            v_metric.markdown(f"""
                <div class="emotion-card st-emotion-1">
                    <div class="metric-label">Facial Vibe</div>
                    <div class="metric-value">{vs.primary_emotion}</div>
                    <div class="metric-delta">{vs.confidence:.1%} match</div>
                </div>
            """, unsafe_allow_html=True)
            
            # Show top 5 blendshapes
            top_bs = dict(sorted(vs.blendshapes.items(), key=lambda x: x[1], reverse=True)[:5])
            bs_chart.write(top_bs)

        # Update Audio Metrics
        if app_state.audio_result:
            as_ = app_state.audio_result
            a_metric.markdown(f"""
                <div class="emotion-card st-emotion-2">
                    <div class="metric-label">Vocal Vibe</div>
                    <div class="metric-value">{as_.primary_emotion}</div>
                    <div class="metric-delta">{as_.confidence:.1%} match</div>
                </div>
            """, unsafe_allow_html=True)
            
            transcription_box.markdown(f"""
                <div class="emotion-card st-emotion-3">
                    <div class="metric-label">Live Transcript</div>
                    <div class="metric-value" style="font-size: 1.1rem; line-height: 1.5; color: var(--accent-glow);">
                        "{as_.transcription}"
                    </div>
                </div>
            """, unsafe_allow_html=True)

        time.sleep(0.1)
