import streamlit as st
import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
import torch
import os
import imutils
import pygame
import mediapipe as mp
from datetime import datetime
from facial_emotion_recognition import EmotionRecognition
import time
import threading
from queue import Queue

# Set page configuration and styling
st.set_page_config(
    page_title="Drowsiness Detection System",
    page_icon="😴",
    layout="wide"
)

# Add custom CSS for improved appearance
st.markdown("""
<style>
    .main {
        padding: 1rem 1rem;
    }
    .stAlert > div {
        padding: 0.3rem 0.5rem;
        margin-bottom: 0.5rem;
    }
    h1, h2, h3 {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
        margin-bottom: 0.5rem;
    }
    .event-box {
        padding: 0.5rem;
        border-radius: 0.3rem;
        margin-bottom: 0.3rem;
        background-color: #FFF9C4;
        border-left: 3px solid #FFC107;
    }
    .stProgress > div > div {
        height: 10px;
    }
    .stSlider > div > div {
        height: 8px;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 0.8rem;
        border-radius: 0.3rem; 
        margin-bottom: 0.5rem;
    }
    .video-container {
        background-color: #000;
        border-radius: 0.5rem;
        overflow: hidden;
    }
    .css-1p0fhdb {
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Initialize threading components
frame_queue = Queue(maxsize=1)
processed_frame_queue = Queue(maxsize=1)
stop_event = threading.Event()

# Initialize pygame for alarm sound
pygame.mixer.init()

# Initialize MediaPipe Face Mesh with optimized settings
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.3,
    static_image_mode=False
)
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Define eye landmarks indexes for MediaPipe Face Mesh
# Left eye indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

# Fix for emotion recognition - create a custom class to override the model loading
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
        st.sidebar.success(f'Emotion Recognition Model Accuracy: {model_dict["accuracy"]:.2f}')
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

def eyeAspectRatio(eye_points):
    # Vertical distances
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Horizontal distance
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

def process_frame(frame, ear_threshold, time_threshold_seconds, fps=30, detection_active=True):
    # Convert frames to time threshold
    time_frames = int(time_threshold_seconds * fps)
    
    # Reduce frame size for processing - improves performance
    frame = imutils.resize(frame, width=480)
    
    # Add status info to the frame
    status_text = "ACTIVE" if detection_active else "PAUSED"
    status_color = (0, 255, 0) if detection_active else (0, 0, 255)
    cv.putText(frame, f"Status: {status_text}", (frame.shape[1] - 120, 30), 
              cv.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
    
    # Convert the image to RGB for MediaPipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    h, w, _ = frame.shape
    
    if detection_active:
        # Process the image with MediaPipe Face Mesh
        results = face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Extract eye landmarks
                left_eye_points = []
                right_eye_points = []
                
                # Process only key eye landmarks for better performance
                for idx in LEFT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    left_eye_points.append((x, y))
                    cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                for idx in RIGHT_EYE:
                    landmark = face_landmarks.landmark[idx]
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    right_eye_points.append((x, y))
                    cv.circle(frame, (x, y), 1, (0, 255, 0), -1)
                
                # Calculate the eye aspect ratio (EAR) for both eyes
                leftEAR = eyeAspectRatio(left_eye_points)
                rightEAR = eyeAspectRatio(right_eye_points)
                
                # Average the EAR for both eyes
                ear = (leftEAR + rightEAR) / 2.0
                st.session_state.ear_value = ear  # Update session state with current EAR
                
                # Draw contours around the eyes - simpler version for better performance
                left_eye_hull = cv.convexHull(np.array(left_eye_points))
                right_eye_hull = cv.convexHull(np.array(right_eye_points))
                cv.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
                cv.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)
                
                # Add text with current EAR value
                cv.putText(frame, f"EAR: {ear:.2f}", (10, 30), 
                          cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Check if the EAR is below the threshold
                if ear < ear_threshold:
                    st.session_state.count += 1
                    
                    # Add progress bar showing time elapsed toward threshold
                    progress = min(1.0, st.session_state.count / time_frames)
                    progress_width = int(progress * 180)
                    cv.rectangle(frame, (10, h - 40), (10 + progress_width, h - 30), (0, 0, 255), -1)
                    cv.rectangle(frame, (10, h - 40), (190, h - 30), (255, 255, 255), 1)
                    
                    if st.session_state.count >= time_frames:
                        if not st.session_state.alarm_triggered:
                            # Play alarm
                            pygame.mixer.music.load('alarm.wav')
                            pygame.mixer.music.play(-1)
                            st.session_state.alarm_triggered = True
                            
                            # Record drowsy event
                            now = datetime.now()
                            st.session_state.drowsy_events.append({
                                'timestamp': now.strftime('%Y-%m-%d %H:%M:%S'),
                                'ear_value': ear
                            })
                            if len(st.session_state.drowsy_events) > 10:
                                st.session_state.drowsy_events = st.session_state.drowsy_events[-10:]
                                
                        cv.putText(frame, "DROWSY!", (10, 60),
                                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        # Add red overlay to indicate drowsiness
                        overlay = frame.copy()
                        cv.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                        frame = cv.addWeighted(overlay, 0.2, frame, 0.8, 0)
                else:
                    st.session_state.count = 0
                    if st.session_state.alarm_triggered:
                        pygame.mixer.music.stop()
                        st.session_state.alarm_triggered = False
                
                # Perform emotion recognition occasionally (every 5 frames)
                if (time.time() * 10) % 50 < 1:  # Only process emotions ~every half second
                    try:
                        emotion_frame = er.recognise_emotion(frame, return_type='BGR')
                        if emotion_frame is not None:
                            frame = emotion_frame
                            # Extract emotion from frame
                            for face_box in frame:
                                if isinstance(face_box, dict) and 'emotion' in face_box:
                                    st.session_state.current_emotion = face_box['emotion']
                    except Exception as e:
                        print(f"Error in emotion recognition: {e}")
    
    # Add timestamp
    cv.putText(frame, datetime.now().strftime('%H:%M:%S'), 
              (w - 80, h - 10), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return frame

def camera_thread(ear_threshold, time_threshold, detection_active):
    """Thread function to capture and process frames"""
    # Initialize camera
    cap = cv.VideoCapture(0)
    
    # Optimize camera properties for performance
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Set lower FPS for better performance
    cap.set(cv.CAP_PROP_FPS, 15)
    
    frame_count = 0
    
    try:
        while not stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every other frame to reduce CPU load
            frame_count += 1
            if frame_count % 2 == 0:  # Skip every other frame
                continue
                
            # Process the frame
            processed = process_frame(
                frame,
                ear_threshold, 
                time_threshold,
                detection_active=detection_active
            )
            
            # Put the processed frame in the queue
            if not processed_frame_queue.full():
                processed_frame_queue.put(processed)
                
            time.sleep(0.01)  # Small delay to prevent CPU hogging
            
    except Exception as e:
        st.error(f"Camera thread error: {e}")
    finally:
        cap.release()

def main():
    # Initialize session state variables
    if 'count' not in st.session_state:
        st.session_state.count = 0
    if 'alarm_triggered' not in st.session_state:
        st.session_state.alarm_triggered = False
    if 'drowsy_events' not in st.session_state:
        st.session_state.drowsy_events = []
    if 'current_emotion' not in st.session_state:
        st.session_state.current_emotion = "Unknown"
    if 'ear_value' not in st.session_state:
        st.session_state.ear_value = 0.0
    if 'last_ear_threshold' not in st.session_state:
        st.session_state.last_ear_threshold = 0.2
    if 'last_time_threshold' not in st.session_state:
        st.session_state.last_time_threshold = 1.0
    if 'last_detection_active' not in st.session_state:
        st.session_state.last_detection_active = True
    if 'thread_started' not in st.session_state:
        st.session_state.thread_started = False

    # Main title - use HTML for better styling
    st.markdown("<h1 style='text-align: center; color: #2E86C1;'>Drowsiness Detection System</h1>", unsafe_allow_html=True)
    st.markdown("<h3 style='text-align: center; font-weight: 400; margin-bottom: 30px;'>Driver Safety Monitoring</h3>", unsafe_allow_html=True)
    
    # Initialize the emotion recognition model
    global er
    if 'er' not in st.session_state:
        with st.spinner("Loading Emotion Recognition Model..."):
            er = CPUEmotionRecognition(device='cpu')
            st.session_state.er = er
    else:
        er = st.session_state.er
    
    # Create sidebar for controls with better styling
    st.sidebar.markdown("<h2 style='text-align: center; margin-bottom: 20px;'>Controls</h2>", unsafe_allow_html=True)
    
    # Eye Aspect Ratio Threshold with better description
    st.sidebar.markdown("### EAR Threshold")
    ear_threshold = st.sidebar.slider(
        "Adjust sensitivity",
        min_value=0.1,
        max_value=0.4,
        value=0.2,
        step=0.01,
        help="Lower value = more sensitive to drowsiness (detects smaller eye closures)"
    )
    
    # Time Threshold with better designed text input
    st.sidebar.markdown("### Time Threshold")
    st.sidebar.markdown("How long eyes must be closed to trigger alert:")
    
    # Better layout for time threshold input
    col1, col2 = st.sidebar.columns([3, 1])
    with col1:
        time_threshold_input = st.text_input(
            "",  # No label, we use the markdown above
            value="1.0",
            key="time_threshold_input"
        )
    with col2:
        st.markdown("<div style='margin-top: 8px;'>seconds</div>", unsafe_allow_html=True)
    
    # Add a clear apply button
    if st.sidebar.button("Apply Settings", type="primary"):
        st.sidebar.success(f"✅ Settings applied successfully!")
    
    # Validate and convert time threshold
    try:
        time_threshold = float(time_threshold_input)
        if time_threshold < 0.1:
            time_threshold = 0.1
            st.sidebar.warning("⚠️ Minimum threshold is 0.1 seconds")
        elif time_threshold > 10.0:
            time_threshold = 10.0
            st.sidebar.warning("⚠️ Maximum threshold is 10 seconds")
    except ValueError:
        time_threshold = 1.0
        st.sidebar.error("❌ Please enter a valid number")
    
    # Detection controls in an expander for cleaner sidebar
    with st.sidebar.expander("Advanced Controls", expanded=True):
        detection_active = st.checkbox("Detection Active", value=True)
        
        # Alarm controls
        if st.button("Stop Alarm", type="secondary"):
            if st.session_state.get('alarm_triggered', False):
                pygame.mixer.music.stop()
                st.session_state.alarm_triggered = False
                st.success("Alarm stopped!")
        
        # Reset events
        if st.button("Clear Events", type="secondary"):
            st.session_state.drowsy_events = []
            st.success("Event history cleared!")
    
    # Create a layout with two columns - better proportions
    col1, col2 = st.columns([7, 3])
    
    # Column 1: Video Feed with better styling
    with col1:
        st.markdown("<div class='video-container'>", unsafe_allow_html=True)
        video_placeholder = st.empty()
        st.markdown("</div>", unsafe_allow_html=True)
        
    # Column 2: Statistics and Events with better styling
    with col2:
        # Current stats
        st.markdown("<h3>Real-time Statistics</h3>", unsafe_allow_html=True)
        
        stats_container = st.container()
        drowsy_events_container = st.container()
    
    # Start the camera thread
    if 'thread_started' not in st.session_state:
        stop_event.clear()
        thread = threading.Thread(
            target=camera_thread, 
            args=(ear_threshold, time_threshold, detection_active),
            daemon=True
        )
        thread.start()
        st.session_state.thread_started = True
    
    # Main UI update loop
    try:
        placeholder = st.empty()
        
        while True:
            # Get the processed frame
            if not processed_frame_queue.empty():
                frame = processed_frame_queue.get()
                
                # Convert to RGB for display
                rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                
                # Display the frame
                video_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
            # Update stats with better styling
            with stats_container:
                st.markdown("<div class='metric-container'>", unsafe_allow_html=True)
                cols = st.columns(2)
                with cols[0]:
                    status = "Active" if detection_active else "Inactive"
                    status_color = "green" if detection_active else "red"
                    st.markdown(f"<span style='font-weight:bold'>Status:</span> <span style='color:{status_color}'>{status}</span>", unsafe_allow_html=True)
                    
                with cols[1]:
                    ear_value = getattr(st.session_state, 'ear_value', 0.0)
                    ear_color = "red" if ear_value < ear_threshold else "green"
                    st.markdown(f"<span style='font-weight:bold'>EAR:</span> <span style='color:{ear_color}'>{ear_value:.2f}</span>", unsafe_allow_html=True)
                
                emotion_cols = st.columns(2)
                with emotion_cols[0]:
                    emotion = getattr(st.session_state, 'current_emotion', "Unknown")
                    st.markdown(f"<span style='font-weight:bold'>Emotion:</span> {emotion}", unsafe_allow_html=True)
                
                with emotion_cols[1]:
                    if ear_value < ear_threshold and detection_active:
                        frames_progress = min(1.0, st.session_state.count / (time_threshold * 15))
                        st.progress(frames_progress)
                    else:
                        st.progress(0.0)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Update drowsy events with better styling
            with drowsy_events_container:
                st.markdown("<h3>Drowsiness Events</h3>", unsafe_allow_html=True)
                events = getattr(st.session_state, 'drowsy_events', [])
                
                if not events:
                    st.info("No drowsiness events recorded")
                else:
                    for event in reversed(events):
                        st.markdown(
                            f"<div class='event-box'><strong>{event['timestamp']}</strong> - EAR: {event['ear_value']:.2f}</div>",
                            unsafe_allow_html=True
                        )
            
            # Short pause to reduce CPU usage and prevent flickering
            time.sleep(0.1)
            
            # Check if settings have changed and restart thread if needed
            if ear_threshold != st.session_state.get('last_ear_threshold', None) or \
               time_threshold != st.session_state.get('last_time_threshold', None) or \
               detection_active != st.session_state.get('last_detection_active', None):
                
                # Stop current thread
                stop_event.set()
                time.sleep(0.5)  # Allow time for thread to stop
                
                # Clear the event and start a new thread
                stop_event.clear()
                thread = threading.Thread(
                    target=camera_thread, 
                    args=(ear_threshold, time_threshold, detection_active),
                    daemon=True
                )
                thread.start()
                
                # Update last known settings
                st.session_state.last_ear_threshold = ear_threshold
                st.session_state.last_time_threshold = time_threshold
                st.session_state.last_detection_active = detection_active
            
            # Break out of loop if Streamlit session ends
            if not placeholder.empty():
                placeholder.empty()
            else:
                break
                
    except Exception as e:
        st.error(f"Error: {e}")
    finally:
        # Make sure we stop the thread when the app is closed
        stop_event.set()

# Run the app
if __name__ == "__main__":
    main()
