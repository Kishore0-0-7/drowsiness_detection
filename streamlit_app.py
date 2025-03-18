import streamlit as st
import cv2
import numpy as np
from scipy.spatial import distance as dist
import sys
from typing import Any, List, Tuple, Dict, Optional
if not hasattr(cv2, "CAP_PROP_FRAME_WIDTH"):
    setattr(cv2, "CAP_PROP_FRAME_WIDTH", 3)
    setattr(cv2, "CAP_PROP_FRAME_HEIGHT", 4)
    setattr(cv2, "FONT_HERSHEY_SIMPLEX", 0)
    setattr(cv2, "COLOR_BGR2RGB", 4)
    setattr(cv2, "VideoCapture", type("VideoCapture", (), {"__call__": lambda *args: object()}))
    for attr in ["putText", "circle", "rectangle", "convexHull", "drawContours", "cvtColor", "addWeighted"]:
        if not hasattr(cv2, attr):
            setattr(cv2, attr, lambda *args, **kwargs: None)
import torch
import os
import imutils
import pygame
import mediapipe as mp
import time
from datetime import datetime
from facial_emotion_recognition import EmotionRecognition

# Configure page
st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add custom CSS for better appearance
st.markdown("""
<style>
    .main { padding: 1rem; }
    h1, h2, h3 { font-family: 'Segoe UI', sans-serif; }
    .stat-box {
        background-color: #3264c6;
        border-radius: 5px;
        padding: 10px 15px;
        margin-bottom: 10px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.12);
    }
    .drowsy-event {
        background-color: #564646;
        border-left: 4px solid #ff7272;
        padding: 8px 12px;
        margin-bottom: 8px;
        border-radius: 0 4px 4px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.1);
    }
    .progress-container {
        height: 15px;
        background-color: #eee;
        border-radius: 10px;
        margin: 10px 0;
    }
    .progress-bar {
        height: 100%;
        border-radius: 10px;
        background-color: #ff5757;
    }
    .video-container {
        background-color: #000;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        border: 1px solid #333;
        margin-bottom: 20px;
    }
    .control-button {
        background-color: #4CAF50;
        color: white;
        padding: 8px 16px;
        border: none;
        border-radius: 4px;
        cursor: pointer;
        margin-right: 8px;
    }
    .stButton>button {
        width: 100%;
        border-radius: 10px;
        padding: 0.3rem 1rem;
        margin-bottom: 0.5rem;
        font-weight: 500;
        transition: all 0.2s ease;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.15);
    }
    div.block-container {padding-top: 1rem;}
    [data-testid="stMetricLabel"] {font-size: 1rem; color: #555;}
    [data-testid="stMetricValue"] {font-size: 1.8rem; font-weight: bold;}
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'count' not in st.session_state:
    st.session_state.count = 0
if 'alarm_triggered' not in st.session_state:
    st.session_state.alarm_triggered = False
if 'ear_value' not in st.session_state:
    st.session_state.ear_value = 0.0
if 'detection_active' not in st.session_state:
    st.session_state.detection_active = True
if 'detection_paused' not in st.session_state:
    st.session_state.detection_paused = False
if 'current_emotion' not in st.session_state:
    st.session_state.current_emotion = "Unknown"
if 'drowsy_events' not in st.session_state:
    st.session_state.drowsy_events = []
if 'earThresh' not in st.session_state:
    st.session_state.earThresh = 0.2
if 'earFrames' not in st.session_state:
    st.session_state.earFrames = 30
if 'fps' not in st.session_state:
    st.session_state.fps = 15
if 'time_threshold' not in st.session_state:
    st.session_state.time_threshold = 1.0
if 'current_alarm_sound' not in st.session_state:
    st.session_state.current_alarm_sound = 'alarm.wav'
if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False
if 'total_drowsy_events' not in st.session_state:
    st.session_state.total_drowsy_events = 0
if 'monitoring_time' not in st.session_state:
    st.session_state.monitoring_time = datetime.now()
if 'show_face_mesh' not in st.session_state:
    st.session_state.show_face_mesh = True
if 'selected_tab' not in st.session_state:
    st.session_state.selected_tab = "Monitor"
if 'reports' not in st.session_state:
    st.session_state.reports = []

# Initialize pygame for alarm sound
pygame.mixer.init()

# Available alarm sounds - can be expanded
alarm_sounds = {
    'Default Alarm': 'alarm.wav',
    'Alarm 2': 'alarm.wav',  # Add more sound files as needed
    'Alarm 3': 'alarm.wav',
}

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Define eye landmarks indexes for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

class CPUEmotionRecognition(EmotionRecognition):
    def __init__(self, device='cpu', gpu_id=0):
        # Skip the parent init which has the problematic torch.load
        super(EmotionRecognition, self).__init__()
        self.device = device
        
        # Import needed for finding the package
        import facial_emotion_recognition
        
        # Add map_location parameter to force CPU
        model_dict = torch.load(os.path.join(os.path.dirname(facial_emotion_recognition.__file__), 
                                'model', 'model.pkl'), 
                                map_location=torch.device('cpu'))
        self.emotions = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Sad', 5: 'Surprise', 6: 'Neutral'}
        
        # Initialize the network
        from facial_emotion_recognition.networks import NetworkV2
        import torch.nn as nn
        from facenet_pytorch import MTCNN
        from torchvision import transforms
        
        self.network = NetworkV2(in_c=1, nl=32, out_f=7).to(self.device)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((48, 48)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        self.mtcnn = MTCNN(keep_all=True, device=self.device)
        
        self.network.load_state_dict(model_dict['network'])
        self.network.eval()

# Initialize emotion recognition with custom class
er = CPUEmotionRecognition(device='cpu')

def eyeAspectRatio(eye_points):
    # Vertical distances
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Horizontal distance
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def stop_alarm():
    if st.session_state.alarm_triggered:
        pygame.mixer.music.stop()
        st.session_state.alarm_triggered = False
        
def calculate_monitoring_duration():
    """Calculate how long the system has been monitoring"""
    now = datetime.now()
    delta = now - st.session_state.monitoring_time
    hours, remainder = divmod(delta.seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}"

def get_eye_color(ear, ear_thresh):
    """Get color based on EAR value compared to threshold"""
    if ear < ear_thresh:
        return (0, 0, 255)  # Red for below threshold
    elif ear < ear_thresh + 0.1:
        return (0, 255, 255)  # Yellow for near threshold
    else:
        return (0, 255, 0)  # Green for well above threshold

def process_frame(frame):
    # Get current parameters from session state
    ear_thresh = st.session_state.earThresh
    frames_threshold = st.session_state.earFrames
    detection_active = st.session_state.detection_active
    detection_paused = st.session_state.detection_paused
    show_face_mesh = st.session_state.show_face_mesh
    
    # Resize frame for consistent processing
    frame = imutils.resize(frame, width=640)
    h, w, _ = frame.shape
    
    # Add status info to the frame
    status_color = (0, 255, 0) if detection_active and not detection_paused else (0, 0, 255)
    cv2.putText(frame, "Status: " + ("ACTIVE" if detection_active and not detection_paused else "PAUSED"), 
               (w-200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    
    if not detection_paused:
        # Convert to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                left_eye_points = []
                right_eye_points = []
                
                for idx in LEFT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    left_eye_points.append((x, y))
                    if show_face_mesh:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                for idx in RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    right_eye_points.append((x, y))
                    if show_face_mesh:
                        cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Calculate eye aspect ratio
                leftEAR = eyeAspectRatio(left_eye_points)
                rightEAR = eyeAspectRatio(right_eye_points)
                ear = (leftEAR + rightEAR) / 2.0
                st.session_state.ear_value = ear  # Update global ear value
                
                # Draw eye contours
                left_eye_hull = cv2.convexHull(np.array(left_eye_points))
                right_eye_hull = cv2.convexHull(np.array(right_eye_points))
                
                # Get eye color based on EAR value
                eye_color = get_eye_color(ear, ear_thresh)
                
                if show_face_mesh:
                    cv2.drawContours(frame, [left_eye_hull], -1, eye_color, 1)
                    cv2.drawContours(frame, [right_eye_hull], -1, eye_color, 1)
                
                # Add EAR value text with dynamic color
                cv2.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, eye_color, 2)
                
                # Add time threshold text
                seconds_threshold = frames_threshold / st.session_state.fps
                cv2.putText(frame, f"Threshold: < {ear_thresh:.2f} for {seconds_threshold:.1f}s", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Check for drowsiness
                if detection_active and ear < ear_thresh:
                    st.session_state.count += 1
                    
                    # Add visual progress indicator
                    progress = min(1.0, st.session_state.count / frames_threshold)
                    progress_width = int(progress * 200)
                    cv2.rectangle(frame, (10, h-50), (10 + progress_width, h-30), (0, 0, 255), -1)
                    cv2.rectangle(frame, (10, h-50), (210, h-30), (255, 255, 255), 1)
                    
                    if st.session_state.count >= frames_threshold:
                        if not st.session_state.alarm_triggered:
                            # Play alarm sound
                            pygame.mixer.music.load(st.session_state.current_alarm_sound)
                            pygame.mixer.music.play(-1)
                            st.session_state.alarm_triggered = True
                            
                            # Record drowsy event
                            now = datetime.now()
                            st.session_state.drowsy_events.append({
                                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                                'ear_value': ear,
                                'emotion': st.session_state.current_emotion
                            })
                            # Keep only the most recent 20 events
                            if len(st.session_state.drowsy_events) > 20:
                                st.session_state.drowsy_events = st.session_state.drowsy_events[-20:]
                                
                            # Update total drowsy events counter
                            st.session_state.total_drowsy_events += 1
                        
                        # Add drowsiness warning
                        cv2.putText(frame, "DROWSINESS DETECTED", (10, 90),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        
                        # Add red overlay
                        overlay = frame.copy()
                        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                        frame = cv2.addWeighted(overlay, 0.2, frame, 0.8, 0)
                else:
                    st.session_state.count = 0
                    if st.session_state.alarm_triggered:
                        pygame.mixer.music.stop()
                        st.session_state.alarm_triggered = False
                
                # Perform emotion recognition
                try:
                    emotion_frame = er.recognise_emotion(frame, return_type='BGR')
                    if emotion_frame is not None:
                        frame = emotion_frame
                        # Extract emotion from frame
                        for face_box in frame:
                            if isinstance(face_box, dict) and 'emotion' in face_box:
                                st.session_state.current_emotion = face_box['emotion']
                except Exception as e:
                    pass
        else:
            # No face detected
            cv2.putText(frame, "No face detected", (int(w/2)-100, int(h/2)), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    
    # Add emotion text
    emotion_color = (255, 0, 255)  # Purple for emotion
    cv2.putText(frame, f"Emotion: {st.session_state.current_emotion}", (10, h-80), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, emotion_color, 2)
    
    # Add timestamp and monitoring duration
    monitoring_duration = calculate_monitoring_duration()
    cv2.putText(frame, f"Time: {datetime.now().strftime('%H:%M:%S')} | Monitoring: {monitoring_duration}", 
               (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return frame

def main():
    # Set up the sidebar
    with st.sidebar:
        st.title("Drowsiness Detection")
        st.image("https://img.icons8.com/fluency/96/000000/sleep.png", width=80)
        
        # Navigation tabs as radio buttons with better styling
        tabs = ["Monitor", "Settings", "Statistics", "Help"]
        selected_tab = st.radio("Navigation", tabs, index=tabs.index(st.session_state.selected_tab) if st.session_state.selected_tab in tabs else 0)
        st.session_state.selected_tab = selected_tab
        
        st.markdown("---")
        
        # Quick controls in sidebar
        st.subheader("Quick Controls")
        
        # Two buttons in one row
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Start/Stop", use_container_width=True):
                st.session_state.detection_active = not st.session_state.detection_active
                if st.session_state.detection_active:
                    st.success("Detection started")
                else:
                    st.error("Detection stopped")
        
        with col2:
            if st.button("Pause/Resume", use_container_width=True):
                st.session_state.detection_paused = not st.session_state.detection_paused
                
        # Stop Alarm
        if st.button("Stop Alarm", use_container_width=True, type="primary" if st.session_state.alarm_triggered else "secondary"):
            stop_alarm()
            st.success("Alarm stopped")
        
        st.markdown("---")
        
        # Status indicators
        st.subheader("System Status")
        st.markdown(f"""
        <div class="stat-box">
            <strong>Detection:</strong> {'✅ Active' if st.session_state.detection_active else '❌ Inactive'}
        </div>
        <div class="stat-box">
            <strong>Status:</strong> {'⏸️ Paused' if st.session_state.detection_paused else '▶️ Running'}
        </div>
        """, unsafe_allow_html=True)
        
        # Add monitoring session time
        st.markdown(f"<div class='stat-box'><strong>Session:</strong> {calculate_monitoring_duration()}</div>", unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Display current parameters
        st.caption("© 2023 Drowsiness Detection System")
    
    # Main content area
    if selected_tab == "Monitor":
        # Main monitoring tab content
        st.header("👁️ Real-time Driver Safety Monitoring")
        
        # Create two columns layout
        col1, col2 = st.columns([3, 1])
        
        with col1:
            # Video container with better styling
            st.markdown("<div class='video-container'>", unsafe_allow_html=True)
            video_frame = st.empty()
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Current EAR and status metrics below video
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric(label="Eye Aspect Ratio", value=f"{st.session_state.ear_value:.2f}", 
                         delta=f"{st.session_state.ear_value - st.session_state.earThresh:.2f}",
                         delta_color="inverse")
            
            with metrics_col2:
                st.metric(label="Emotion", value=st.session_state.current_emotion)
                
            with metrics_col3:
                st.metric(label="Drowsy Events", value=st.session_state.total_drowsy_events)
        
        with col2:
            # Last few drowsy events
            st.subheader("Recent Drowsy Events")
            if not st.session_state.drowsy_events:
                st.info("No drowsiness events recorded")
            else:
                for event in reversed(st.session_state.drowsy_events[:5]):
                    st.markdown(f"""
                    <div class="drowsy-event">
                        <strong>{event['timestamp'].split(' ')[1]}</strong><br>
                        EAR: {event['ear_value']:.2f} | {event.get('emotion', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                if len(st.session_state.drowsy_events) > 5:
                    if st.button("Show All Events", key="show_all"):
                        st.session_state.selected_tab = "Statistics"
            
            # Quick settings/toggles
            st.subheader("Quick Settings")
            
            # Toggle face mesh visibility
            st.checkbox("Show Face Mesh", value=st.session_state.show_face_mesh, 
                        key="face_mesh", on_change=lambda: setattr(st.session_state, "show_face_mesh", not st.session_state.show_face_mesh))
            
            # Quick threshold adjustment
            st.slider("EAR Threshold", min_value=0.15, max_value=0.3, 
                     value=st.session_state.earThresh, step=0.01, key="quick_ear_thresh")
    
    elif selected_tab == "Settings":
        # Settings tab content
        st.header("⚙️ System Settings")
        
        # Create tabs for different setting categories
        settings_tabs = st.tabs(["Detection", "Appearance", "Audio", "Advanced"])
        
        with settings_tabs[0]:
            st.subheader("Detection Settings")
            
            # EAR Threshold slider
            ear_thresh = st.slider(
                "Eye Aspect Ratio Threshold",
                min_value=0.1,
                max_value=0.4,
                value=st.session_state.earThresh,
                step=0.01,
                help="Lower values are more sensitive to eye closure"
            )
            st.session_state.earThresh = ear_thresh
            
            # Time threshold input
            time_threshold = st.number_input(
                "Time Threshold (seconds)",
                min_value=0.1,
                max_value=10.0,
                value=st.session_state.time_threshold,
                step=0.1,
                help="How long eyes must be closed to trigger an alert"
            )
            # Convert seconds to frames
            st.session_state.earFrames = int(time_threshold * st.session_state.fps)
            st.session_state.time_threshold = time_threshold
            
            # Face detection confidence
            face_confidence = st.slider(
                "Face Detection Confidence",
                min_value=0.1,
                max_value=0.9,
                value=0.5,
                step=0.1,
                help="Higher values require more confidence for face detection"
            )
            
            # Reset detection count button
            if st.button("Reset Detection Count"):
                st.session_state.count = 0
                st.session_state.drowsy_events = []
                st.session_state.total_drowsy_events = 0
                st.session_state.monitoring_time = datetime.now()
                st.success("Detection counts and events have been reset")
                
        with settings_tabs[1]:
            st.subheader("Appearance Settings")
            
            # Dark mode toggle
            dark_mode = st.toggle("Dark Mode", value=st.session_state.dark_mode)
            st.session_state.dark_mode = dark_mode
            
            # UI theme selection
            ui_theme = st.selectbox(
                "UI Theme",
                options=["Classic", "Modern", "Minimal"],
                index=1,
                help="Select the interface theme"
            )
            
            # Show/hide face mesh
            show_mesh = st.checkbox("Show Face Mesh", value=st.session_state.show_face_mesh)
            st.session_state.show_face_mesh = show_mesh
            
            # Display alert style
            alert_style = st.radio(
                "Alert Style",
                options=["Flash Screen", "Warning Text", "Both"],
                index=2,
                help="How to display drowsiness alerts"
            )
        
        with settings_tabs[2]:
            st.subheader("Audio Settings")
            
            # Alarm sound selection
            selected_sound = st.selectbox(
                "Alarm Sound",
                options=list(alarm_sounds.keys()),
                index=0,
                help="Select the alarm sound to play when drowsiness is detected"
            )
            st.session_state.current_alarm_sound = alarm_sounds[selected_sound]
            
            # Volume slider
            volume = st.slider(
                "Alarm Volume",
                min_value=0,
                max_value=100,
                value=80,
                step=5,
                help="Adjust the volume of the alarm"
            )
            
            # Test alarm button
            if st.button("Test Alarm"):
                pygame.mixer.music.load(st.session_state.current_alarm_sound)
                pygame.mixer.music.play(0)  # Play once
                st.info("Playing test alarm...")
                
            # Mute toggle
            mute = st.toggle("Mute All Sounds", value=False)
            
        with settings_tabs[3]:
            st.subheader("Advanced Settings")
            
            # FPS setting
            fps = st.number_input(
                "Target FPS",
                min_value=5,
                max_value=30,
                value=st.session_state.fps,
                step=1,
                help="Target frames per second (higher values use more CPU)"
            )
            st.session_state.fps = fps
            
            # Advanced warning time
            warning_time = st.number_input(
                "Warning Time (seconds)",
                min_value=0.0,
                max_value=5.0,
                value=0.5,
                step=0.1,
                help="How early to show a warning before full alert"
            )
            
            # Camera selection
            camera_id = st.selectbox(
                "Camera Source",
                options=["0: Default Camera", "1: Secondary Camera"],
                index=0,
                help="Select which camera to use"
            )
            
            # Export settings button
            if st.button("Export Settings"):
                st.download_button(
                    label="Download Settings",
                    data=str({
                        "earThresh": st.session_state.earThresh,
                        "time_threshold": st.session_state.time_threshold,
                        "fps": st.session_state.fps,
                        "alarm_sound": st.session_state.current_alarm_sound
                    }),
                    file_name="drowsiness_settings.json",
                    mime="application/json"
                )
            
            # Import/Export Section
            st.subheader("Import/Export Settings")
            
            col1, col2 = st.columns(2)
            with col1:
                # Export all data
                export_json = export_data()
                st.download_button(
                    label="Export All Data",
                    data=export_json,
                    file_name="drowsiness_data.json",
                    mime="application/json",
                    help="Export all settings, events, and statistics"
                )
                
            with col2:
                # Import settings
                uploaded_file = st.file_uploader("Import Settings", type="json", help="Import settings from a JSON file")
                if uploaded_file is not None:
                    settings_json = uploaded_file.read().decode("utf-8")
                    success, message = import_settings(settings_json)
                    if success:
                        st.success(message)
                    else:
                        st.error(message)
    
    elif selected_tab == "Statistics":
        # Statistics tab content
        st.header("📊 Monitoring Statistics")
        
        # Create summary metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Monitoring Time", calculate_monitoring_duration())
        with col2:
            st.metric("Drowsy Events", st.session_state.total_drowsy_events)
        with col3:
            if st.session_state.drowsy_events:
                time_since_last = datetime.now() - datetime.strptime(st.session_state.drowsy_events[-1]['timestamp'], '%Y-%m-%d %H:%M:%S')
                minutes_since = time_since_last.total_seconds() // 60
                st.metric("Time Since Last Event", f"{int(minutes_since)} minutes")
            else:
                st.metric("Time Since Last Event", "N/A")
        
        # Stats tabs for different views
        stats_tabs = st.tabs(["Event Log", "Reports", "Visualization"])
        
        with stats_tabs[0]:
            # Create detailed event log
            st.subheader("Drowsy Event Log")
            if not st.session_state.drowsy_events:
                st.info("No drowsiness events recorded yet")
            else:
                # Add filters
                col1, col2 = st.columns(2)
                with col1:
                    filter_emotion = st.multiselect("Filter by Emotion", 
                                                  options=list(set([event.get('emotion', 'Unknown') for event in st.session_state.drowsy_events])),
                                                  default=[])
                with col2:
                    sort_order = st.radio("Sort Order", ["Newest First", "Oldest First"], horizontal=True)
                
                # Apply filters
                filtered_events = st.session_state.drowsy_events
                if filter_emotion:
                    filtered_events = [event for event in filtered_events if event.get('emotion', 'Unknown') in filter_emotion]
                    
                # Apply sort
                if sort_order == "Oldest First":
                    filtered_events = list(reversed(filtered_events))
                    
                # Display events
                for event in filtered_events:
                    st.markdown(f"""
                    <div class="drowsy-event">
                        <strong>{event['timestamp']}</strong><br>
                        EAR: {event['ear_value']:.2f} | Emotion: {event.get('emotion', 'Unknown')}
                    </div>
                    """, unsafe_allow_html=True)
                    
                # Export events button
                if st.button("Export Events to CSV"):
                    import pandas as pd
                    import io
                    
                    # Convert events to dataframe
                    df = pd.DataFrame(st.session_state.drowsy_events)
                    
                    # Convert to CSV
                    csv = df.to_csv(index=False)
                    
                    # Offer download
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name="drowsiness_events.csv", 
                        mime="text/csv"
                    )
                
                # Clear events button
                if st.button("Clear All Events"):
                    st.session_state.drowsy_events = []
                    st.session_state.total_drowsy_events = 0
                    st.success("Event log cleared")
        
        with stats_tabs[1]:
            # Reports section
            st.subheader("Analysis Reports")
            
            # Generate new report button
            if st.button("Generate New Report"):
                report, message = generate_report()
                if report:
                    st.success(message)
                else:
                    st.warning(message)
            
            # Display existing reports
            if not st.session_state.reports:
                st.info("No reports generated yet. Click 'Generate New Report' to create one.")
            else:
                # Select report to view
                report_timestamps = [f"Report {i+1}: {r['timestamp']}" for i, r in enumerate(st.session_state.reports)]
                selected_report = st.selectbox("Select Report to View", report_timestamps)
                
                if selected_report:
                    report_idx = int(selected_report.split(':')[0].replace('Report ', '')) - 1
                    report = st.session_state.reports[report_idx]
                    
                    # Display report in an expandable section
                    with st.expander("Report Summary", expanded=True):
                        st.markdown(f"""
                        ### Report Generated: {report['timestamp']}
                        
                        **Total Drowsy Events:** {report['total_events']}  
                        **Average EAR Value:** {report['avg_ear']:.3f}  
                        **Minimum EAR Value:** {report['min_ear']:.3f}  
                        **Most Common Emotion:** {report['most_common_emotion']}
                        
                        #### Settings During Report
                        - EAR Threshold: {report['settings']['ear_threshold']}
                        - Time Threshold: {report['settings']['time_threshold']} seconds
                        """)
                    
                    # Emotion distribution
                    st.subheader("Emotion Distribution")
                    emotion_data = report['emotion_distribution']
                    if emotion_data:
                        # Convert to dataframe for charting
                        import pandas as pd
                        emotion_df = pd.DataFrame({
                            'Emotion': list(emotion_data.keys()),
                            'Count': list(emotion_data.values())
                        })
                        
                        st.bar_chart(emotion_df.set_index('Emotion'))
                    
                    # Hourly distribution
                    st.subheader("Hourly Distribution")
                    hour_data = report['hourly_distribution']
                    if hour_data:
                        # Convert to dataframe with all 24 hours
                        import pandas as pd
                        hours_df = pd.DataFrame({
                            'Hour': list(range(0, 24)),
                            'Count': [hour_data.get(str(h), 0) for h in range(0, 24)]
                        })
                        
                        st.bar_chart(hours_df.set_index('Hour'))
                    
                    # Export report button
                    st.download_button(
                        label="Export Report",
                        data=str(report),
                        file_name=f"drowsiness_report_{report['timestamp'].replace(':', '-').replace(' ', '_')}.json",
                        mime="application/json"
                    )
        
        with stats_tabs[2]:
            # Data visualization tab
            st.subheader("Data Visualization")
            
            if not st.session_state.drowsy_events:
                st.info("No data available for visualization")
            else:
                # Import visualization libraries
                import pandas as pd
                import numpy as np
                import plotly.express as px
                import plotly.graph_objects as go
                
                # Convert events to DataFrame
                df = pd.DataFrame(st.session_state.drowsy_events)
                
                # Convert timestamps to datetime
                df['datetime'] = pd.to_datetime(df['timestamp'])
                df['hour'] = df['datetime'].dt.hour
                
                # Plot options
                plot_type = st.radio("Select Plot Type", 
                                    ["EAR Values Over Time", "Events by Hour", "Emotion Distribution"],
                                    horizontal=True)
                
                if plot_type == "EAR Values Over Time":
                    fig = px.line(df, x='datetime', y='ear_value', 
                                 title='Eye Aspect Ratio Over Time',
                                 labels={'ear_value': 'EAR Value', 'datetime': 'Time'})
                    
                    # Add threshold line
                    fig.add_hline(y=st.session_state.earThresh, line_dash="dash", 
                                 line_color="red", annotation_text="Threshold")
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Events by Hour":
                    hour_counts = df['hour'].value_counts().sort_index()
                    
                    fig = px.bar(x=hour_counts.index, y=hour_counts.values,
                                title='Drowsy Events by Hour of Day',
                                labels={'x': 'Hour of Day', 'y': 'Number of Events'})
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                elif plot_type == "Emotion Distribution":
                    if 'emotion' in df.columns:
                        emotion_counts = df['emotion'].value_counts()
                        
                        fig = px.pie(values=emotion_counts.values, names=emotion_counts.index,
                                    title='Distribution of Emotions During Drowsy Events')
                        
                        st.plotly_chart(fig, use_container_width=True)
                    else:
                        st.info("No emotion data available")
    
    elif selected_tab == "Help":
        # Help tab content
        st.header("❓ Help & Information")
        
        # Add tabs for different help sections
        help_tabs = st.tabs(["Getting Started", "FAQ", "About"])
        
        with help_tabs[0]:
            st.subheader("Getting Started")
            st.markdown("""
            ### How to Use the Drowsiness Detection System
            
            1. **Position your face** in view of the camera
            2. **Adjust the EAR threshold** if needed (lower values are more sensitive)
            3. **Set the time threshold** to determine how long eyes must be closed to trigger an alert
            4. **Start monitoring** and the system will alert you when drowsiness is detected
            
            ### Tips for Best Results
            
            - Ensure good lighting for accurate face detection
            - Position the camera at eye level
            - Adjust the thresholds based on your personal eye characteristics
            - Test the system before relying on it during driving
            """)
            
            # Quick setup guide with buttons
            st.subheader("Quick Setup")
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Use Default Settings", use_container_width=True):
                    st.session_state.earThresh = 0.2
                    st.session_state.time_threshold = 1.0
                    st.session_state.earFrames = int(st.session_state.time_threshold * st.session_state.fps)
                    st.success("Default settings applied")
            with col2:
                if st.button("Use Sensitive Settings", use_container_width=True):
                    st.session_state.earThresh = 0.25
                    st.session_state.time_threshold = 0.8
                    st.session_state.earFrames = int(st.session_state.time_threshold * st.session_state.fps)
                    st.success("Sensitive settings applied")
        
        with help_tabs[1]:
            st.subheader("Frequently Asked Questions")
            
            # Use expanders for FAQ items
            with st.expander("What is the Eye Aspect Ratio (EAR)?"):
                st.markdown("""
                The Eye Aspect Ratio (EAR) is a measure of how open your eyes are. 
                
                It's calculated based on the ratio of the height to the width of the eye. When your eyes are fully open, the EAR value is higher. When your eyes close, the EAR value drops significantly.
                
                The system uses this value to detect when your eyes are closing due to drowsiness.
                """)
                
            with st.expander("How accurate is the emotion detection?"):
                st.markdown("""
                The emotion detection uses a neural network model to classify facial expressions into seven basic emotions: Angry, Disgust, Fear, Happy, Sad, Surprise, and Neutral.
                
                While it works well in good lighting conditions with clear facial features, it may not be 100% accurate in all situations. The emotion detection is provided as an additional feature and is not critical to the drowsiness detection functionality.
                """)
                
            with st.expander("Can I use this system while wearing glasses?"):
                st.markdown("""
                Yes, the system can generally work with glasses, but with some limitations:
                
                - Non-reflective glasses work better than glasses with glare
                - Thick frames may interfere with eye landmark detection
                - Sunglasses will prevent proper eye detection
                
                For best results, use clear glasses with minimal reflection.
                """)
                
            with st.expander("Why is the system not detecting my face?"):
                st.markdown("""
                If the system is having trouble detecting your face, try the following:
                
                1. Ensure adequate lighting - face should be clearly visible
                2. Position yourself directly in front of the camera
                3. Remove any obstacles blocking your face
                4. Restart the application if the issue persists
                """)
        
        with help_tabs[2]:
            st.subheader("About This System")
            
            st.markdown("""
            ### Drowsiness Detection System
            
            Version 1.1.0
            
            This application is designed to detect signs of drowsiness in drivers or operators by monitoring their eye state in real-time. It uses advanced computer vision techniques and machine learning to track facial landmarks and analyze eye closure patterns.
            
            **Key Technologies:**
            - MediaPipe for facial landmark detection
            - OpenCV for image processing
            - Emotion recognition with neural networks
            - Streamlit for the user interface
            
            **Contact:**
            For support or feedback, please contact support@drowsinessdetection.com
            """)
            
            # System information
            st.subheader("System Information")
            system_info = {
                "Python Version": sys.version.split(' ')[0],
                "OpenCV Version": cv2.__version__,
                "MediaPipe Version": mp.__version__,
                "Streamlit Version": st.__version__,
                "Operating System": sys.platform
            }
            
            for key, value in system_info.items():
                st.text(f"{key}: {value}")
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    # Check if camera opened successfully
    if not cap.isOpened():
        st.error("Error: Could not open camera. Please check your webcam connection.")
        return
    
    # Camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    frame_skip = 0  # For performance optimization
    
    # Main video loop - process frames from webcam
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break
            
            # Skip frames for better performance based on target FPS
            frame_skip += 1
            if frame_skip % max(1, int(30 / st.session_state.fps)) != 0:
                continue
            
            # Process the frame for drowsiness detection
            processed_frame = process_frame(frame)
            
            # Convert to RGB for display
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            
            # Display the processed frame
            video_frame.image(rgb_frame, use_column_width=True)
            
            # Short sleep to reduce CPU usage
            time.sleep(0.03)
            
    except Exception as e:
        st.error(f"Error processing video feed: {e}")
    finally:
        cap.release()

def export_data():
    """Export all session data as JSON"""
    import json
    
    export_data = {
        "settings": {
            "earThresh": st.session_state.earThresh,
            "time_threshold": st.session_state.time_threshold,
            "fps": st.session_state.fps,
            "alarm_sound": st.session_state.current_alarm_sound,
            "show_face_mesh": st.session_state.show_face_mesh,
            "dark_mode": st.session_state.dark_mode
        },
        "events": st.session_state.drowsy_events,
        "statistics": {
            "total_events": st.session_state.total_drowsy_events,
            "session_start": st.session_state.monitoring_time.isoformat()
        }
    }
    
    return json.dumps(export_data, indent=2)

def import_settings(settings_json):
    """Import settings from JSON"""
    import json
    
    try:
        data = json.loads(settings_json)
        if "settings" in data:
            settings = data["settings"]
            for key, value in settings.items():
                if key in st.session_state:
                    setattr(st.session_state, key, value)
            
            # Update frames based on time threshold
            st.session_state.earFrames = int(st.session_state.time_threshold * st.session_state.fps)
            return True, "Settings imported successfully"
        else:
            return False, "Invalid settings format: 'settings' key not found"
    except Exception as e:
        return False, f"Error importing settings: {str(e)}"

def generate_report():
    """Generate a summary report of drowsiness events"""
    from collections import Counter
    import pandas as pd
    
    if not st.session_state.drowsy_events:
        return None, "No drowsy events to generate report"
    
    # Extract data for analysis
    timestamps = [event['timestamp'] for event in st.session_state.drowsy_events]
    ear_values = [event['ear_value'] for event in st.session_state.drowsy_events]
    emotions = [event.get('emotion', 'Unknown') for event in st.session_state.drowsy_events]
    
    # Basic statistics
    avg_ear = sum(ear_values) / len(ear_values)
    min_ear = min(ear_values)
    emotion_counts = Counter(emotions)
    most_common_emotion = emotion_counts.most_common(1)[0][0]
    
    # Create hourly distribution
    hours = [datetime.strptime(ts, '%Y-%m-%d %H:%M:%S').hour for ts in timestamps]
    hour_counts = Counter(hours)
    
    # Create report
    report = {
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "total_events": len(st.session_state.drowsy_events),
        "avg_ear": avg_ear,
        "min_ear": min_ear,
        "most_common_emotion": most_common_emotion,
        "emotion_distribution": dict(emotion_counts),
        "hourly_distribution": dict(hour_counts),
        "settings": {
            "ear_threshold": st.session_state.earThresh,
            "time_threshold": st.session_state.time_threshold
        }
    }
    
    # Store the report
    st.session_state.reports.append(report)
    
    return report, "Report generated successfully"

if __name__ == "__main__":
    main()