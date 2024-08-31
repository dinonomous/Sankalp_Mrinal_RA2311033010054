import cv2
import numpy as np
from django.http import StreamingHttpResponse
from django.shortcuts import render
from django.views.decorators import gzip
import mediapipe as mp

# Initialize MediaPipe Face Detection and Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
mp_face_recognition = mp.solutions.face_detection.FaceDetection(min_detection_confidence=0.7)
face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.7)

# Define constants and colors
BLINK_THRESHOLD = 5.0
FACE_CONFIDENCE_THRESHOLD = 0.7
DATASET_PATH = "face_dataset"
RECOGNITION_THRESHOLD = 0.7
color = (0, 255, 255)
white_color = (255, 255, 255)
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]

def euclidean_distance(point1, point2):
    x1, y1 = point1.ravel()
    x2, y2 = point2.ravel()
    return np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

def iris_position(iris_center, right_point, left_point):
    center_to_right_dist = euclidean_distance(iris_center, right_point)
    total_distance = euclidean_distance(right_point, left_point)
    ratio = center_to_right_dist / total_distance
    if ratio <= 0.42:
        return "right", ratio
    elif 0.42 < ratio <= 0.57:
        return "center", ratio
    else:
        return "left", ratio

def generate_frames():
    cap = cv2.VideoCapture(0)  # Capture from the first webcam

    while True:
        success, frame = cap.read()
        if not success:
            break

        # Process the frame for face detection and recognition
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_results = face_detection.process(rgb_image)

        if face_results.detections:
            for detection in face_results.detections:
                face_confidence = detection.score[0]
                if face_confidence >= FACE_CONFIDENCE_THRESHOLD:
                    # Draw face bounding box and landmarks
                    face_box = detection.location_data.relative_bounding_box
                    h, w, c = frame.shape
                    x_min = int(face_box.xmin * w)
                    y_min = int(face_box.ymin * h)
                    width = int(face_box.width * w)
                    height = int(face_box.height * h)

                    cv2.rectangle(frame, (x_min, y_min), (x_min + width, y_min + height), color, 2)

                    # Additional processing can be added here (e.g., blink detection)

        # Encode frame as JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@gzip.gzip_page
def live_feed(request):
    return StreamingHttpResponse(generate_frames(), content_type='multipart/x-mixed-replace; boundary=frame')

def index(request):
    return render(request, 'index.html')
