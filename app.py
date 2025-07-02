from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
import tensorflow as tf
from tensorflow import keras
import threading
import time
import json
from datetime import datetime
import base64
import os

app = Flask(__name__)

# Global variables
camera = None
model = None
drowsy_count = 0
alert_threshold = 10  # Number of consecutive drowsy frames to trigger alert
detection_active = False
last_detection_time = None
detection_history = []

class DrowsinessDetector:
    def __init__(self, model_path='final_improved_model.h5'):
        self.model = None
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.load_model(model_path)
        self.img_size = 128
        
    def load_model(self, model_path):
        try:
            if os.path.exists(model_path):
                self.model = keras.models.load_model(model_path)
                print(f"Model loaded successfully from {model_path}")
            else:
                print(f"Model file {model_path} not found. Please ensure the model is trained and saved.")
                self.model = None
        except Exception as e:
            print(f"Error loading model: {e}")
            self.model = None
    
    def preprocess_face(self, face_img):
        """Preprocess face image for model prediction"""
        try:
            # Resize to model input size
            face_resized = cv2.resize(face_img, (self.img_size, self.img_size))
            # Normalize pixel values
            face_normalized = face_resized.astype(np.float32) / 255.0
            # Add batch dimension
            face_batch = np.expand_dims(face_normalized, axis=0)
            return face_batch
        except Exception as e:
            print(f"Error preprocessing face: {e}")
            return None
    
    def detect_drowsiness(self, frame):
        """Detect drowsiness in the given frame"""
        global drowsy_count, last_detection_time, detection_history
        
        if self.model is None:
            return frame, "Model not loaded", 0.0
        
        # Convert to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        prediction_text = "No face detected"
        confidence = 0.0
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Extract face region
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                # Preprocess face for model
                processed_face = self.preprocess_face(face_roi)
                
                if processed_face is not None:
                    # Make prediction
                    prediction = self.model.predict(processed_face, verbose=0)[0][0]
                    confidence = float(prediction)
                    
                    # Determine drowsiness state (assuming 0 = drowsy, 1 = alert)
                    if prediction < 0.5:
                        prediction_text = "DROWSY"
                        drowsy_count += 1
                        color = (0, 0, 255)  # Red
                    else:
                        prediction_text = "ALERT"
                        drowsy_count = max(0, drowsy_count - 1)
                        color = (0, 255, 0)  # Green
                    
                    # Update detection history
                    current_time = datetime.now()
                    detection_history.append({
                        'timestamp': current_time.strftime('%H:%M:%S'),
                        'status': prediction_text,
                        'confidence': confidence
                    })
                    
                    # Keep only last 50 detections
                    if len(detection_history) > 50:
                        detection_history.pop(0)
                    
                    last_detection_time = current_time
                    
                    # Add text overlay
                    cv2.putText(frame, f"{prediction_text}", (x, y-10), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
                    cv2.putText(frame, f"Conf: {confidence:.2f}", (x, y+h+20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add drowsiness counter
        cv2.putText(frame, f"Drowsy Count: {drowsy_count}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return frame, prediction_text, confidence

# Initialize detector
detector = DrowsinessDetector()

def get_camera():
    global camera
    if camera is None:
        camera = cv2.VideoCapture(0)
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    return camera

def generate_frames():
    global detection_active
    
    camera = get_camera()
    
    while True:
        success, frame = camera.read()
        if not success:
            break
        
        if detection_active:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect drowsiness
            frame, status, confidence = detector.detect_drowsiness(frame)
            
            # Add timestamp
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            cv2.putText(frame, timestamp, (10, frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame)
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

@app.route('/start_detection', methods=['POST'])
def start_detection():
    global detection_active, drowsy_count
    detection_active = True
    drowsy_count = 0
    return jsonify({'status': 'started', 'message': 'Detection started successfully'})

@app.route('/stop_detection', methods=['POST'])
def stop_detection():
    global detection_active
    detection_active = False
    return jsonify({'status': 'stopped', 'message': 'Detection stopped'})

@app.route('/get_stats')
def get_stats():
    global drowsy_count, last_detection_time, detection_history
    
    # Calculate statistics
    total_detections = len(detection_history)
    drowsy_detections = len([d for d in detection_history if d['status'] == 'DROWSY'])
    alert_detections = total_detections - drowsy_detections
    
    drowsy_percentage = (drowsy_detections / total_detections * 100) if total_detections > 0 else 0
    
    return jsonify({
        'drowsy_count': drowsy_count,
        'total_detections': total_detections,
        'drowsy_detections': drowsy_detections,
        'alert_detections': alert_detections,
        'drowsy_percentage': round(drowsy_percentage, 2),
        'last_detection': last_detection_time.strftime('%H:%M:%S') if last_detection_time else 'N/A',
        'detection_active': detection_active,
        'alert_triggered': drowsy_count >= alert_threshold
    })

@app.route('/get_history')
def get_history():
    global detection_history
    return jsonify({'history': detection_history[-10:]})  # Return last 10 detections

@app.route('/reset_stats', methods=['POST'])
def reset_stats():
    global drowsy_count, detection_history
    drowsy_count = 0
    detection_history = []
    return jsonify({'status': 'reset', 'message': 'Statistics reset successfully'})

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    global alert_threshold
    
    if request.method == 'POST':
        data = request.get_json()
        alert_threshold = data.get('alert_threshold', alert_threshold)
        return jsonify({'status': 'updated', 'alert_threshold': alert_threshold})
    
    return jsonify({'alert_threshold': alert_threshold})

if __name__ == '__main__':
    print("Starting Drowsiness Detection App...")
    print("Make sure your model file 'final_improved_model.h5' is in the same directory")
    print("Open http://localhost:5000 in your web browser")
    app.run(debug=True, threaded=True)