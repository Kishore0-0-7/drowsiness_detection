import serial
import cv2
import imutils
import dlib
from scipy.spatial import distance as dist
from imutils import face_utils
import pygame

# Initialize Pygame for sound alerts
pygame.mixer.init()
sound = pygame.mixer.Sound("alarm.wav")

# Constants
earThresh = 0.3  # Distance between vertical eye coordinates threshold
earFrames = 48   # Consecutive frames for eye closure
shapePredictor = "shape_predictor_68_face_landmarks.dat"  # Path to shape predictor file

# Initialize serial communication (adjust the port name as needed)
arduino_port = "COM3"  # Change this to the correct port
ser = serial.Serial(arduino_port, 9600, timeout=1)

def eyeAspectRatio(eye):
    # Calculate eye aspect ratio
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

count = 0
drowsiness_active = False

# Initialize face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(shapePredictor)

# Get the coordinates of left and right eye
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Initialize webcam
cam = cv2.VideoCapture(0)

while True:
    _, frame = cam.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    rects = detector(gray, 0)

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]

        leftEAR = eyeAspectRatio(leftEye)
        rightEAR = eyeAspectRatio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0

        if ear < earThresh:
            count += 1
            if count >= earFrames:
                if not drowsiness_active:
                    print("DROWSINESS DETECTED")
                    sound.play()
                    drowsiness_active = True

                # Continuously send 'dt' command to Arduino while drowsiness is detected
                ser.write(b'st\n')
                print("DROWSINESS FOUND")
        else:
            count = 0
            drowsiness_active = False
            # Send 'nd' command to Arduino if no drowsiness is detected
            ser.write(b'nt\n')
            print("NO DROWSINESS FOUND")

        # Draw eye contours on the frame
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 0, 255), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 0, 255), 1)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

# Clean up
cam.release()
cv2.destroyAllWindows()
ser.close()  # Close the serial connection
pygame.mixer.quit()