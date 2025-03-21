import streamlit as st
import av
import numpy as np
import cv2
import mediapipe as mp
from scipy.spatial import distance as dist
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import streamlit.components.v1 as components

# JavaScript-based audio functions for alarms
def play_alarm():
    """Plays an alarm sound using JavaScript in Streamlit."""
    alarm_script = """
    <script>
        var audio = document.getElementById("alarm_sound");
        if (!audio) {
            audio = document.createElement("audio");
            audio.id = "alarm_sound";
            audio.src = "https://www.soundjay.com/button/beep-07.wav";
            audio.loop = true;
            document.body.appendChild(audio);
        }
        audio.play();
    </script>
    """
    components.html(alarm_script, height=0)

def stop_alarm():
    """Stops the alarm sound using JavaScript in Streamlit."""
    stop_script = """
    <script>
        var audio = document.getElementById("alarm_sound");
        if (audio) {
            audio.pause();
            audio.currentTime = 0;
        }
    </script>
    """
    components.html(stop_script, height=0)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Define eye landmarks indexes for MediaPipe Face Mesh
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eye_aspect_ratio(eye_points):
    """Calculates the Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

# Streamlit UI
st.set_page_config(page_title="🚗 Drowsiness Detection", layout="wide")

st.title("🚗 Real-Time Drowsiness Detection System")
st.markdown("""
This app detects drowsiness in real-time using a webcam.  
If your **Eye Aspect Ratio (EAR)** goes below a threshold for too long, an alarm will trigger.  

### **🔹 How to use:**
1. **Click 'Enable Alarm' before starting detection** (required for sound to work in browsers).
2. **Grant webcam access when prompted.**
3. **Look at the screen** and blink normally.
4. **If drowsy, the alarm will sound.**
""")

# Buttons to enable/disable alarm
if "alarm_enabled" not in st.session_state:
    st.session_state["alarm_enabled"] = False
if "drowsy" not in st.session_state:
    st.session_state["drowsy"] = False

col1, col2 = st.columns(2)
with col1:
    if st.button("🔊 Enable Alarm"):
        st.session_state["alarm_enabled"] = True
        st.success("✅ Alarm system activated.")

with col2:
    if st.button("🔕 Disable Alarm"):
        st.session_state["alarm_enabled"] = False
        stop_alarm()
        st.warning("⏹️ Alarm system disabled.")

# Show "Drowsiness Detected" message if active
if st.session_state["drowsy"]:
    st.error("🚨 **DROWSINESS DETECTED! WAKE UP!** 🚨")
else:
    st.success("✅ **You are alert.**")

# Video Processor Class
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.ear_threshold = 0.2  # EAR threshold for drowsiness
        self.frames_threshold = 30  # Frames before triggering alarm
        self.count = 0
        self.alarm_triggered = False

    def recv(self, frame):
        """Processes live webcam frames for drowsiness detection."""
        image = frame.to_ndarray(format="bgr24")
        h, w, _ = image.shape

        # Convert frame to RGB for MediaPipe processing
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_image)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in LEFT_EYE]
                right_eye = [(int(face_landmarks.landmark[i].x * w), int(face_landmarks.landmark[i].y * h)) for i in RIGHT_EYE]

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                # Draw eye landmarks
                color = (0, 255, 0) if ear > self.ear_threshold else (0, 0, 255)
                cv2.putText(image, f"EAR: {ear:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                cv2.drawContours(image, [np.array(left_eye)], -1, color, 1)
                cv2.drawContours(image, [np.array(right_eye)], -1, color, 1)

                # Drowsiness detection logic
                if ear < self.ear_threshold:
                    self.count += 1
                    if self.count >= self.frames_threshold:
                        if not self.alarm_triggered and st.session_state.get("alarm_enabled", False):
                            st.session_state["drowsy"] = True
                            play_alarm()  # Plays alarm sound in browser
                            self.alarm_triggered = True
                else:
                    self.count = 0
                    if self.alarm_triggered:
                        st.session_state["drowsy"] = False
                        stop_alarm()  # Stops alarm sound
                        self.alarm_triggered = False

        return av.VideoFrame.from_ndarray(image, format="bgr24")

# Start webcam stream
webrtc_streamer(
    key="drowsiness-detection",
    video_processor_factory=VideoProcessor,
    async_processing=True,  # Fixes threading issues
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)