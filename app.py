import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode
import av
import cv2
import numpy as np
import time
import queue
import threading
from core.vision_engine import VisionEngine
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
    @import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&display=swap');

    :root {
        --bg-deep: #050505;
        --accent-glow: #fbbf24;
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

    .emotion-card {
        background: var(--card-bg);
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(12px);
        padding: 24px;
        border-radius: 20px;
        margin-bottom: 20px;
        transition: all 0.3s ease;
        animation: fadeInUp 0.6s ease-out both;
        text-align: center;
    }

    .emotion-card:hover {
        border-color: var(--accent-glow);
        transform: translateY(-3px);
        box-shadow: 0 10px 30px rgba(0,0,0,0.5);
    }

    .emotion-label {
        color: var(--text-dim);
        font-size: 0.7rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 2px;
        margin-bottom: 8px;
    }

    .emotion-value {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text-main);
        line-height: 1.1;
        margin-top: 4px;
    }

    .emotion-conf {
        color: var(--accent-glow);
        font-size: 0.85rem;
        font-weight: 600;
        margin-top: 6px;
    }

    .section-label {
        font-size: 0.65rem;
        font-weight: 700;
        letter-spacing: 3px;
        text-transform: uppercase;
        color: #4b5563;
        margin-bottom: 10px;
        margin-top: 6px;
    }

    .bs-row {
        display: flex;
        align-items: center;
        margin-bottom: 8px;
        gap: 10px;
    }
    .bs-name {
        font-size: 0.72rem;
        color: #9ca3af;
        width: 160px;
        flex-shrink: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }
    .bs-bar-bg {
        flex: 1;
        height: 6px;
        background: rgba(255,255,255,0.05);
        border-radius: 4px;
        overflow: hidden;
    }
    .bs-bar-fill {
        height: 100%;
        border-radius: 4px;
        background: linear-gradient(90deg, #6366f1, #8b5cf6);
    }
    .bs-val {
        font-size: 0.7rem;
        color: #6b7280;
        width: 34px;
        text-align: right;
        flex-shrink: 0;
    }

    .element-container iframe, .stWebRtcStreamer {
        border-radius: 24px;
        border: 2px solid rgba(255,255,255,0.05);
        overflow: hidden;
        box-shadow: 0 20px 50px rgba(0,0,0,0.8);
    }

    [data-testid="stCheckbox"] label {
        font-size: 0.8rem !important;
        color: #6b7280 !important;
    }

    @keyframes fadeInUp {
        from { opacity: 0; transform: translateY(30px); }
        to { opacity: 1; transform: translateY(0); }
    }
    @keyframes fadeInDown {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }

    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }
</style>

<div class="main-title">CharacterLab</div>
<div class="sub-title">Real-Time Emotion Mirror · Vision Engine</div>
""", unsafe_allow_html=True)

# --- Shared State ---
class AppState:
    flip_camera = True
    vision_result = None

app_state = AppState()

# --- Engine Initialization ---
@st.cache_resource
def load_engines():
    return VisionEngine()

try:
    vision_engine = load_engines()
except Exception as e:
    st.error(f"Error loading engines: {e}")
    st.stop()

# --- Emotion maps ---
EMOTION_EMOJI = {
    "Joy": "😄", "Sadness": "😢", "Anger": "😠",
    "Surprise": "😲", "Disgust": "🤢", "Fear": "😨", "Neutral": "😐",
}
EMOTION_COLOR = {
    "Joy":      "#fbbf24",
    "Sadness":  "#60a5fa",
    "Anger":    "#f87171",
    "Surprise": "#a78bfa",
    "Disgust":  "#34d399",
    "Fear":     "#fb923c",
    "Neutral":  "#9ca3af",
}

# --- Callbacks ---
def video_frame_callback(frame: av.VideoFrame) -> av.VideoFrame:
    img = frame.to_ndarray(format="bgr24")

    if app_state.flip_camera:
        img = cv2.flip(img, 1)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    timestamp_ms = int(time.time() * 1000)

    try:
        vision_state = vision_engine.process_frame(rgb_img, timestamp_ms)
        if vision_state:
            app_state.vision_result = vision_state
            emotion = vision_state.primary_emotion
            hex_color = EMOTION_COLOR.get(emotion, "#6366f1")
            bgr = tuple(int(hex_color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))
            label = f"{EMOTION_EMOJI.get(emotion, '')} {emotion}  {vision_state.confidence:.0%}"
            cv2.putText(img, label, (20, 44), cv2.FONT_HERSHEY_DUPLEX, 1.0, bgr, 2, cv2.LINE_AA)
    except Exception as e:
        print(f"Vision error: {e}")

    return av.VideoFrame.from_ndarray(img, format="bgr24")

# --- Layout ---
col_vid, col_stats = st.columns([3, 1], gap="large")

with col_vid:
    app_state.flip_camera = st.checkbox("Mirror mode", value=True)
    webrtc_ctx = webrtc_streamer(
        key="emotion-mirror",
        mode=WebRtcMode.SENDRECV,
        video_frame_callback=video_frame_callback,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_stats:
    st.markdown('<div class="section-label">Face</div>', unsafe_allow_html=True)
    emotion_card = st.empty()
    st.markdown('<div class="section-label">Top Blendshapes</div>', unsafe_allow_html=True)
    bs_container = st.empty()

    emotion_card.markdown("""
        <div class="emotion-card">
            <div class="emotion-label">Detected Emotion</div>
            <div class="emotion-value">—</div>
            <div class="emotion-conf">Waiting for camera…</div>
        </div>
    """, unsafe_allow_html=True)

# --- Update Loop ---
if webrtc_ctx.state.playing:
    while True:
        vs = app_state.vision_result
        if vs:
            emotion = vs.primary_emotion
            emoji = EMOTION_EMOJI.get(emotion, "")
            color = EMOTION_COLOR.get(emotion, "#6366f1")

            emotion_card.markdown(f"""
                <div class="emotion-card" style="border-color:{color}33;">
                    <div class="emotion-label">Detected Emotion</div>
                    <div class="emotion-value" style="color:{color};">{emoji} {emotion}</div>
                    <div class="emotion-conf">{vs.confidence:.1%} confidence</div>
                </div>
            """, unsafe_allow_html=True)

            top_bs = sorted(vs.blendshapes.items(), key=lambda x: x[1], reverse=True)[:8]
            bars_html = ""
            for name, val in top_bs:
                pct = int(val * 100)
                bars_html += f"""
                <div class="bs-row">
                    <span class="bs-name">{name}</span>
                    <div class="bs-bar-bg"><div class="bs-bar-fill" style="width:{pct}%"></div></div>
                    <span class="bs-val">{pct}%</span>
                </div>"""
            bs_container.markdown(bars_html, unsafe_allow_html=True)

        time.sleep(0.1)
