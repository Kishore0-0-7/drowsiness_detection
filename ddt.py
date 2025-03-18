from scipy.spatial import distance as dist
import numpy as np
import imutils
import cv2 as cv
import pygame
import torch
import os
import mediapipe as mp
from facial_emotion_recognition import EmotionRecognition


pygame.mixer.init()
pygame.mixer.music.load('alarm.wav')

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

# Initialize variables
count = 0
alarm_triggered = False
earThresh = 0.2  # Threshold for eye aspect ratio (adjusted for MediaPipe)
earFrames = 30  # Number of frames for which eyes need to be below threshold

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

# Start video capture
cam = cv.VideoCapture(0)

# For key variable initialization
key = 0

while True:
    # Read frame from camera
    success, frame = cam.read()
    if not success:
        print("Failed to read frame from camera")
        break
        
    frame = imutils.resize(frame, width=600)
    
    # Convert the image to RGB for MediaPipe
    rgb_frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    h, w, _ = frame.shape
    
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
            
            # Draw contours around the eyes
            left_eye_hull = cv.convexHull(np.array(left_eye_points))
            right_eye_hull = cv.convexHull(np.array(right_eye_points))
            cv.drawContours(frame, [left_eye_hull], -1, (0, 0, 255), 1)
            cv.drawContours(frame, [right_eye_hull], -1, (0, 0, 255), 1)
            
            # Add text with current EAR value
            cv.putText(frame, f"EAR: {ear:.2f}", (10, 60), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Check if the EAR is below the threshold
            if ear < earThresh:
                count += 1
                
                # If the eyes have been closed for the specified number of frames
                if count >= earFrames:
                    if not alarm_triggered:
                        pygame.mixer.music.play(-1)  # Play alarm in loop
                        alarm_triggered = True
                    
                    cv.putText(frame, "DROWSINESS DETECTED", (10, 30),
                              cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print("DROWSINESS DETECTED")
            
            else:
                count = 0
                if alarm_triggered:
                    pygame.mixer.music.stop()  # Stop the alarm
                    alarm_triggered = False
                print("DROWSINESS Not Detected")
            
            # Perform emotion recognition
            frame = er.recognise_emotion(frame, return_type='BGR')
    
    # Display the frame
    cv.imshow("Frame", frame)
    key = cv.waitKey(1) & 0xFF
    
    # Exit the loop if 'q' is pressed
    if key == ord("q"):
        break

# Release the video capture and close all windows
cam.release()
cv.destroyAllWindows()
