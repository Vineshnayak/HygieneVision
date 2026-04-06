import pymongo
from pymongo import MongoClient
import streamlit as st
import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import time
from hygiene_engine import HygieneEngine
from PIL import Image

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="HygieneVision",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- CUSTOM CSS -----------------
# Implementing a Glassmorphism & Modern Dark Theme
css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

[data-testid="stAppViewContainer"] {
    background: radial-gradient(circle at top, #1e293b 0%, #0f172a 100%);
    color: #e2e8f0;
}

[data-testid="stSidebar"] {
    background-color: rgba(15, 23, 42, 0.7);
    backdrop-filter: blur(12px);
    border-right: 1px solid rgba(255,255,255,0.05);
}

.stMetric {
    background: rgba(30, 41, 59, 0.6) !important;
    border-radius: 12px;
    padding: 16px !important;
    border: 1px solid rgba(255,255,255,0.08);
    backdrop-filter: blur(8px);
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
    transition: transform 0.2s ease, border-color 0.2s ease;
}

.stMetric:hover {
    transform: translateY(-2px);
    border-color: rgba(56, 189, 248, 0.5);
}

div.stButton > button:first-child {
    background: linear-gradient(135deg, #0ea5e9, #3b82f6);
    color: white;
    border-radius: 8px;
    border: none;
    padding: 10px 24px;
    font-weight: 600;
    transition: all 0.3s ease;
}

div.stButton > button:first-child:hover {
    background: linear-gradient(135deg, #0284c7, #2563eb);
    transform: scale(1.02);
    box-shadow: 0 0 15px rgba(59, 130, 246, 0.5);
}

.alert-card {
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.2), rgba(220, 38, 38, 0.1));
    border: 2px solid rgba(239, 68, 68, 0.8);
    border-radius: 10px;
    padding: 20px;
    color: #fca5a5;
    font-size: 24px;
    font-weight: bold;
    text-align: center;
    text-transform: uppercase;
    letter-spacing: 2px;
    box-shadow: 0 0 20px rgba(239, 68, 68, 0.4);
    animation: pulse 1.5s infinite;
}

@keyframes pulse {
    0% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
    50% { box-shadow: 0 0 40px rgba(239, 68, 68, 0.8); }
    100% { box-shadow: 0 0 20px rgba(239, 68, 68, 0.4); }
}

.history-entry {
    background: rgba(30, 41, 59, 0.8);
    padding: 12px;
    border-radius: 6px;
    margin-bottom: 8px;
    border-left: 4px solid #ef4444;
}
</style>
"""
st.markdown(css, unsafe_allow_html=True)


# ----------------- SESSION STATE -----------------
@st.cache_resource
def load_engine():
    return HygieneEngine(model_path="best.pt")

@st.cache_resource
def get_mongo_collection():
    try:
        # Connect to isolated DB on port 27019
        client = MongoClient("mongodb://localhost:27019/", serverSelectionTimeoutMS=2000)
        db = client["HygieneVision_db"]
        return db["alert_history"]
    except Exception as e:
        st.error(f"MongoDB connection failed: {e}")
        return None

if 'last_log_time' not in st.session_state:
    st.session_state.last_log_time = None
if 'last_log_violations' not in st.session_state:
    st.session_state.last_log_violations = None

engine = load_engine()
collection = get_mongo_collection()


# ----------------- UI STRUCTURE -----------------
st.title("🛡️ HygieneVision Monitor")
st.markdown("<p style='color: #94a3b8; font-size: 1.1rem;'>Real-time AI surveillance protecting your food standards effortlessly.</p>", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("Control Panel")
    mode = st.radio("Select Input Mode:", ["Live Webcam", "Image Upload", "Video Upload"], index=0)
    
    st.markdown("---")
    st.subheader("Settings")
    staff_name = st.text_input("Active Staff Member", value="Unknown")
    conf_thresh = st.slider("Confidence Threshold", 0.1, 0.9, 0.35, 0.05)
    
    st.markdown("---")
    st.markdown("### 🔔 Total Alert History (MongoDB)")
    # Fetch directly from Database
    if collection is not None:
        recent_history = list(collection.find().sort("timestamp", -1).limit(10))
        if not recent_history:
            st.markdown("<p style='color: #64748b;'>No alerts found in database yet.</p>", unsafe_allow_html=True)
        else:
            for entry in recent_history:
                st.markdown(f"""
                <div class="history-entry">
                    <span style="font-size: 0.8rem; color: #94a3b8;">{entry['timestamp']} | <b>{entry.get('staff_name', 'Unknown')}</b></span><br>
                    <b>{", ".join(entry.get('violations_array', entry.get('violations', []))).replace('_', ' ').title()}</b>
                </div>
                """, unsafe_allow_html=True)
                
            if st.button("Clear DB History"):
                collection.delete_many({})
                st.rerun()
    else:
        st.error("Cannot load history. Database offline.")

# ----------------- MAIN LOGIC -----------------

def log_alert(alerts, input_mode):
    if not alerts or collection is None: return
    now = datetime.now()
    now_str = now.strftime("%Y-%m-%d %H:%M:%S")
    alert_str = ", ".join(alerts)
    
    # Simple 10-second debounce based on latest log to prevent DB spam
    if st.session_state.last_log_time is not None:
        if st.session_state.last_log_violations == alert_str:
            time_diff = (now - st.session_state.last_log_time).total_seconds()
            if time_diff < 10:
                return

    # Create the MongoDB Document
    document = {
        "timestamp": now_str,
        "violations_string": alert_str,
        "violations_array": alerts,
        "input_mode": input_mode,
        "staff_name": staff_name,
        "resolved": False
    }
    
    # Save directly to MongoDB
    collection.insert_one(document)
    
    # Update local debounce tracker
    st.session_state.last_log_time = now
    st.session_state.last_log_violations = alert_str

if mode == "Live Webcam":
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.markdown("### Live Feed")
        run_camera = st.checkbox("START CAMERA", value=False)
        frame_placeholder = st.empty()
    
    with col2:
        st.markdown("### Status")
        status_placeholder = st.empty()
        fps_placeholder = st.empty()

    if run_camera:
        # User explicitly requested mac camera default 0; we use 0
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open the local camera.")
        else:
            status_placeholder.markdown("<div style='color: #22c55e; font-weight:bold;'>● System Active</div>", unsafe_allow_html=True)
            prev_time = time.time()
            
            while run_camera:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Camera connection lost.")
                    break
                
                # Resize for consistency (matches older script)
                frame = cv2.resize(frame, (1280, 720))
                
                # Inference
                annotated_frame, active_alerts, dets = engine.process_frame(frame, conf_threshold=conf_thresh, draw=True)
                
                # Handling Alerts
                if active_alerts:
                    log_alert(active_alerts, "Live Webcam")
                    joined = ", ".join(active_alerts).replace('_', ' ').upper()
                    status_placeholder.markdown(f"<div class='alert-card'>🚨 {joined} 🚨</div>", unsafe_allow_html=True)
                else:
                    status_placeholder.markdown("<div style='color: #22c55e; font-weight:bold; font-size: 1.2rem; padding: 10px; border: 1px solid #22c55e; border-radius: 8px; text-align: center;'>ALL CLEAR</div>", unsafe_allow_html=True)

                # Convert BGR to RGB for Streamlit rendering
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frame_placeholder.image(rgb_frame, channels="RGB", use_container_width=True)
                
                # FPS Calculation
                current_time = time.time()
                fps = 1 / (current_time - prev_time)
                prev_time = current_time
                fps_placeholder.metric("Performance", f"{fps:.1f} FPS")

            cap.release()
    else:
        frame_placeholder.info("Click 'START CAMERA' to begin live real-time analysis.")
        status_placeholder.write("Camera Offline.")


elif mode == "Image Upload":
    st.markdown("### Image Inspector")
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, 1)
        
        with st.spinner("Analyzing high-resolution image..."):
            annotated_img, active_alerts, dets = engine.process_frame(img, conf_threshold=conf_thresh, draw=True)
        
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
        
        if active_alerts:
            log_alert(active_alerts, "Image Upload")
            joined = ", ".join(active_alerts).replace('_', ' ').title()
            st.error(f"Hygiene Violations Detected: {joined}")
        else:
            st.success("No critical violations detected. Standard is upheld.")


elif mode == "Video Upload":
    st.markdown("### Video Audit")
    st.info("Video upload processing requires saving the file and processing it frame by frame. For massive files, performance may vary.")
    uploaded_video = st.file_uploader("Upload a video for batch analysis...", type=["mp4", "avi", "mov"])
    
    if uploaded_video is not None:
        if st.button("Process Video"):
            # Save to temporary path for cv2 reader
            temp_path = f"tmp_video_{int(time.time())}.mp4"
            with open(temp_path, "wb") as f:
                f.write(uploaded_video.read())
            
            cap_vid = cv2.VideoCapture(temp_path)
            frame_placeholder = st.empty()
            prog_bar = st.progress(0)
            
            total_frames = int(cap_vid.get(cv2.CAP_PROP_FRAME_COUNT))
            current_frame = 0
            
            while cap_vid.isOpened():
                ret, frame = cap_vid.read()
                if not ret: break
                
                annotated_img, active_alerts, dets = engine.process_frame(frame, conf_threshold=conf_thresh, draw=True)
                if active_alerts:
                    log_alert(active_alerts, "Video Upload")

                frame_placeholder.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), use_container_width=True)
                
                current_frame += 1
                if total_frames > 0:
                    prog_bar.progress(min(100, int((current_frame / total_frames) * 100)))
            
            cap_vid.release()
            import os
            try: os.remove(temp_path)
            except: pass
            
            st.success("Video processing complete!")
