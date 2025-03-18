import streamlit as st
import cv2
import numpy as np
from scipy.spatial import distance as dist
import sys
import os
import time
from datetime import datetime
from typing import Any, List, Tuple, Dict, Optional
import torch
import imutils
import mediapipe as mp

# Check if running in a headless environment
HEADLESS = "DISPLAY" not in os.environ and os.environ.get("XDG_SESSION_TYPE") != "x11"

# Initialize pygame or use alternative sound method
try:
    import pygame
    if not HEADLESS:
        pygame.mixer.init()  # Initialize sound only if an audio device is present
    else:
        print("No audio device detected. Skipping pygame.mixer.init()")
except Exception as e:
    print(f"Error initializing pygame.mixer: {e}")
    pygame = None

# Alternative sound library for headless environments
try:
    from playsound import playsound
except ImportError:
    playsound = None

# Alarm sound function
def play_alarm():
    """Plays an alarm sound using pygame or playsound."""
    if pygame and not HEADLESS:
        pygame.mixer.music.load("alarm.wav")
        pygame.mixer.music.play(-1)  # Loop indefinitely
    elif playsound:
        playsound("alarm.wav")
    else:
        print("No valid sound library found. Alarm cannot be played.")

def stop_alarm():
    """Stops the alarm sound."""
    if pygame and pygame.mixer.get_busy():
        pygame.mixer.music.stop()

# Configure Streamlit Page
st.set_page_config(
    page_title="Drowsiness Detection",
    page_icon="😴",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
if 'earThresh' not in st.session_state:
    st.session_state.earThresh = 0.2
if 'earFrames' not in st.session_state:
    st.session_state.earFrames = 30
if 'fps' not in st.session_state:
    st.session_state.fps = 15
if 'time_threshold' not in st.session_state:
    st.session_state.time_threshold = 1.0

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
    """Calculates Eye Aspect Ratio (EAR)."""
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    C = dist.euclidean(eye_points[0], eye_points[3])
    return (A + B) / (2.0 * C)

def process_frame(frame):
    """Processes a frame to detect drowsiness based on EAR."""
    ear_thresh = st.session_state.earThresh
    frames_threshold = st.session_state.earFrames
    detection_active = st.session_state.detection_active
    detection_paused = st.session_state.detection_paused

    frame = imutils.resize(frame, width=640)
    h, w, _ = frame.shape

    if not detection_paused:
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                left_eye_points = [(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in LEFT_EYE]
                right_eye_points = [(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in RIGHT_EYE]

                leftEAR = eye_aspect_ratio(left_eye_points)
                rightEAR = eye_aspect_ratio(right_eye_points)
                ear = (leftEAR + rightEAR) / 2.0
                st.session_state.ear_value = ear  

                if detection_active and ear < ear_thresh:
                    st.session_state.count += 1

                    if st.session_state.count >= frames_threshold:
                        if not st.session_state.alarm_triggered:
                            play_alarm()
                            st.session_state.alarm_triggered = True
                else:
                    st.session_state.count = 0
                    if st.session_state.alarm_triggered:
                        stop_alarm()
                        st.session_state.alarm_triggered = False

    return frame

def main():
    """Main Streamlit app loop."""
    st.title("👀 Drowsiness Detection System")
    
    if st.button("Start/Stop Monitoring"):
        st.session_state.detection_active = not st.session_state.detection_active
    
    if st.button("Stop Alarm"):
        stop_alarm()

    st.markdown("---")
    video_frame = st.empty()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Error: Could not open camera.")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame from camera")
                break

            processed_frame = process_frame(frame)
            rgb_frame = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            video_frame.image(rgb_frame, use_column_width=True)

            time.sleep(0.03)

    except Exception as e:
        st.error(f"Error processing video feed: {e}")
    finally:
        cap.release()

if __name__ == "__main__":
    main()
