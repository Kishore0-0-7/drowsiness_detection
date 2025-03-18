from flask import Flask, render_template, Response, jsonify, request
import cv2 as cv
import numpy as np
from scipy.spatial import distance as dist
import torch
import os
import imutils
import pygame
import mediapipe as mp
import time
import threading
import base64
from datetime import datetime
from facial_emotion_recognition import EmotionRecognition

app = Flask(__name__)

# Initialize global variables
count = 0
alarm_triggered = False
earThresh = 0.2  # Threshold for eye aspect ratio (adjusted for MediaPipe)
earFrames = 30  # Number of frames for which eyes need to be below threshold
fps = 30  # Estimated frames per second
detection_active = True
current_emotion = "Unknown"
ear_value = 0.0
last_drowsy_time = None
drowsy_events = []  # Store drowsy events with timestamps
detection_paused = False
current_alarm_sound = 'alarm.wav'  # Default alarm sound

# Initialize alarm sound settings
pygame.mixer.init()
pygame.mixer.music.load(current_alarm_sound)

# Available alarm sounds
alarm_sounds = {
    'alarm1': 'alarm.wav',
    'alarm2': 'alarm.wav',  # Add more sound files as needed
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
mp_drawing_styles = mp.solutions.drawing_styles

# Define eye landmarks indexes for MediaPipe Face Mesh
# Left eye indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
# Right eye indices
RIGHT_EYE = [33, 160, 158, 133, 153, 144]

def eyeAspectRatio(eye_points):
    # Vertical distances
    A = dist.euclidean(eye_points[1], eye_points[5])
    B = dist.euclidean(eye_points[2], eye_points[4])
    # Horizontal distance
    C = dist.euclidean(eye_points[0], eye_points[3])
    ear = (A + B) / (2.0 * C)
    return ear

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
        print(f'[*] Accuracy: {model_dict["accuracy"]}')
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

# Initialize emotion recognition with our custom class
er = CPUEmotionRecognition(device='cpu')

def stop_alarm():
    global alarm_triggered
    if alarm_triggered:
        pygame.mixer.music.stop()
        alarm_triggered = False

def generate_frames():
    global count, alarm_triggered, detection_active, current_emotion, ear_value, last_drowsy_time, drowsy_events, detection_paused
    cam = cv.VideoCapture(0)  # Use 0 for default camera
    
    while True:
        success, frame = cam.read()
        if not success:
            print("Failed to read frame from camera")
            break
            
        frame = imutils.resize(frame, width=600)
        
        # Add status info to the frame
        cv.putText(frame, "Status: " + ("ACTIVE" if detection_active and not detection_paused else "PAUSED"), 
                  (450, 30), cv.FONT_HERSHEY_SIMPLEX, 0.5, 
                  (0, 255, 0) if detection_active and not detection_paused else (0, 0, 255), 2)
        
        # Convert the image to RGB for MediaPipe
        rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        h, w, _ = frame.shape
        
        if not detection_paused:
            # Process the image with MediaPipe Face Mesh
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
                    ear_value = ear  # Update global ear value
                    
                    # Draw contours around the eyes
                    left_eye_hull = cv.convexHull(np.array(left_eye_points))
                    right_eye_hull = cv.convexHull(np.array(right_eye_points))
                    cv.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
                    cv.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)
                    
                    # Add text with current EAR value
                    cv.putText(frame, f"EAR: {ear:.2f}", (10, 60), 
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Check if the EAR is below the threshold
                    if detection_active and ear < earThresh:
                        count += 1
                        if count >= earFrames:
                            if not alarm_triggered:
                                pygame.mixer.music.play(-1)
                                alarm_triggered = True
                                last_drowsy_time = datetime.now()
                                # Record drowsy event
                                drowsy_events.append({
                                    'timestamp': last_drowsy_time.strftime('%Y-%m-%d %H:%M:%S'),
                                    'ear_value': ear
                                })
                            cv.putText(frame, "DROWSINESS DETECTED", (10, 30),
                                        cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            # Add red overlay to indicate drowsiness
                            overlay = frame.copy()
                            cv.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                            frame = cv.addWeighted(overlay, 0.2, frame, 0.8, 0)
                    else:
                        count = 0
                        if alarm_triggered:
                            pygame.mixer.music.stop()
                            alarm_triggered = False

                    # Perform emotion recognition and update global emotion value
                    try:
                        emotion_frame = er.recognise_emotion(frame, return_type='BGR')
                        if emotion_frame is not None:
                            frame = emotion_frame
                            # Extract emotion from frame
                            for face_box in frame:
                                if isinstance(face_box, dict) and 'emotion' in face_box:
                                    current_emotion = face_box['emotion']
                    except Exception as e:
                        print(f"Error in emotion recognition: {e}")

        # Add emotion text
        cv.putText(frame, f"Emotion: {current_emotion}", (10, 90), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
                
        # Add timestamp
        cv.putText(frame, datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
                  (10, h - 20), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        ret, buffer = cv.imencode('.jpg', frame)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/toggle_detection', methods=['POST'])
def toggle_detection():
    global detection_active
    detection_active = not detection_active
    return jsonify({"status": "active" if detection_active else "inactive"})

@app.route('/stop_alarm', methods=['POST'])
def stop_alarm_route():
    stop_alarm()
    return jsonify({"status": "alarm stopped"})

@app.route('/toggle_pause', methods=['POST'])
def toggle_pause():
    global detection_paused
    detection_paused = not detection_paused
    return jsonify({"status": "paused" if detection_paused else "resumed"})

@app.route('/set_threshold', methods=['POST'])
def set_threshold():
    global earThresh
    data = request.get_json()
    if 'threshold' in data:
        try:
            earThresh = float(data['threshold'])
            return jsonify({"status": "success", "threshold": earThresh})
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid threshold value"})
    return jsonify({"status": "error", "message": "No threshold provided"})

@app.route('/set_time_threshold', methods=['POST'])
def set_time_threshold():
    global earFrames
    data = request.get_json()
    if 'timeThreshold' in data:
        try:
            # Convert seconds to frames (approximate)
            seconds = float(data['timeThreshold'])
            earFrames = int(seconds * fps)
            return jsonify({"status": "success", "timeThreshold": seconds, "frames": earFrames})
        except ValueError:
            return jsonify({"status": "error", "message": "Invalid time threshold value"})
    return jsonify({"status": "error", "message": "No time threshold provided"})

@app.route('/set_alarm_sound', methods=['POST'])
def set_alarm_sound():
    global current_alarm_sound
    data = request.get_json()
    if 'sound' in data and data['sound'] in alarm_sounds:
        current_alarm_sound = alarm_sounds[data['sound']]
        try:
            pygame.mixer.music.load(current_alarm_sound)
            return jsonify({"status": "success", "sound": data['sound']})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Failed to load sound: {str(e)}"})
    return jsonify({"status": "error", "message": "Invalid sound selection"})

@app.route('/get_alarm_sounds', methods=['GET'])
def get_alarm_sounds():
    return jsonify({
        "sounds": list(alarm_sounds.keys()),
        "current": current_alarm_sound
    })

@app.route('/get_stats', methods=['GET'])
def get_stats():
    global count, earFrames
    # Calculate seconds from frames
    time_threshold = round(earFrames / fps, 1)
    
    return jsonify({
        "detection_active": detection_active,
        "detection_paused": detection_paused,
        "ear_value": ear_value,
        "current_emotion": current_emotion,
        "threshold": earThresh,
        "timeThreshold": time_threshold,
        "count": count,
        "earFrames": earFrames,
        "currentAlarmSound": current_alarm_sound,
        "drowsy_events": drowsy_events[-10:] if drowsy_events else []
    })

@app.route('/take_snapshot', methods=['POST'])
def take_snapshot():
    # This is a placeholder - in a real app, you'd capture the current frame
    # and save it to a file or database
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return jsonify({"status": "success", "timestamp": timestamp})

@app.route('/reset_events', methods=['POST'])
def reset_events():
    global drowsy_events
    drowsy_events = []
    return jsonify({"status": "success"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')








