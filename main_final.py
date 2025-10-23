import cv2
import mediapipe as mp
import numpy as np
import pyttsx3
import csv
import datetime
import time
import threading
from collections import deque
import os
import platform
import logging

# Set up logging to handle errors better
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Try to import winsound for Windows beeps, fallback for other platforms
try:
    import winsound
    HAS_WINSOUND = True
except ImportError:
    HAS_WINSOUND = False

class DriverMonitoringSystem:
    def __init__(self):
        # Initialize MediaPipe solutions
        self.mp_face_mesh = mp.solutions.face_mesh
        self.mp_hands = mp.solutions.hands
        self.mp_pose = mp.solutions.pose
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # FaceMesh for facial landmarks
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Hands for hand detection
        self.hands = self.mp_hands.Hands(
            model_complexity=0,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Pose for body and face direction detection
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Text-to-speech engine
        try:
            self.tts_engine = pyttsx3.init()
            self.tts_engine.setProperty('rate', 150)
            self.tts_available = True
        except Exception as e:
            logging.warning(f"TTS engine failed to initialize: {e}")
            self.tts_available = False
        
        # Camera setup
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open webcam")
            
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        # DEBUG FIX 1: Correct EAR threshold for proper eye landmarks
        self.EAR_THRESHOLD = 0.20  # Changed from 0.28 to 0.20
        self.YAW_THRESHOLD = 20  # degrees
        self.HAND_OVERLAP_THRESHOLD = 0.15  # 15% area overlap
        
        # Face direction thresholds
        self.LEFT_THRESHOLD = -0.05
        self.RIGHT_THRESHOLD = 0.05
        self.FACE_DIRECTION_CONFIDENCE_THRESHOLD = 0.3
        
        # Face detection failure thresholds
        self.FACE_NOT_DETECTED_THRESHOLD = 3.0  # seconds
        self.ZERO_EAR_THRESHOLD = 0.01  # Minimum EAR value to consider face detected
        self.ZERO_YAW_THRESHOLD = 0.1   # Minimum Yaw value to consider face detected
        
        # Timing and state tracking
        self.eyes_closed_start = None
        self.distracted_start = None
        self.hand_near_start = None
        self.face_not_detected_start = None
        self.last_vocal_alert = 0
        self.last_hand_alert = 0  # ADD THIS LINE
        self.vocal_alert_interval = 10  # seconds
        
        # Smoothing for yaw angles and face direction
        self.yaw_buffer = deque(maxlen=5)
        self.direction_buffer = deque(maxlen=5)
        
        # State tracking
        self.current_state = "Concentrated"
        self.looking_direction = "Center"
        self.face_direction = "Forward"
        self.hand_near_face = False
        self.face_detected = True
        
        # DEBUG FIX 2: Add cooldown tracking to prevent rapid state changes
        self.last_state_change = time.time()
        self.state_cooldown = 0.5  # 500ms cooldown between state changes
        
        # Logging setup
        self.create_logs_folder()
        self.session_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.frame_log_file = f"logs/driver_session_{self.session_timestamp}.csv"
        self.event_log_file = f"logs/driver_events_{self.session_timestamp}.csv"
        
        # File handles for logging
        self.frame_log_handle = None
        self.event_log_handle = None
        self.frame_writer = None
        self.event_writer = None
        
        # Initialize log files
        self.init_log_files()
        
        # Threading for non-blocking TTS
        self.tts_lock = threading.Lock()
        self.is_speaking = False
        
        # System state
        self.running = True
        
        print("Driver Monitoring System initialized!")
        print(f"Frame log: {self.frame_log_file}")
        print(f"Event log: {self.event_log_file}")
    
    def create_logs_folder(self):
        """Create logs directory if it doesn't exist"""
        if not os.path.exists("logs"):
            os.makedirs("logs")
    
    def init_log_files(self):
        """Initialize CSV log files with headers"""
        try:
            # Frame log - keep file open for the session
            self.frame_log_handle = open(self.frame_log_file, 'w', newline='', buffering=1)
            self.frame_writer = csv.writer(self.frame_log_handle)
            self.frame_writer.writerow(['timestamp', 'frame_idx', 'EAR', 'Yaw', 'FaceDirection', 'State'])
            
            # Event log - keep file open for the session
            self.event_log_handle = open(self.event_log_file, 'w', newline='', buffering=1)
            self.event_writer = csv.writer(self.event_log_handle)
            self.event_writer.writerow(['timestamp', 'frame_idx', 'event', 'EAR', 'Yaw', 'FaceDirection'])
            
        except Exception as e:
            logging.error(f"Failed to initialize log files: {e}")
            raise
    
    def log_frame(self, frame_idx, ear, yaw, face_direction, state):
        """Log per-frame data"""
        try:
            if self.frame_writer and self.running:
                timestamp = datetime.datetime.now().isoformat()
                self.frame_writer.writerow([timestamp, frame_idx, ear, yaw, face_direction, state])
        except Exception as e:
            logging.error(f"Frame logging error: {e}")
    
    def log_event(self, frame_idx, event, ear, yaw, face_direction):
        """Log event data"""
        try:
            if self.event_writer and self.running:
                timestamp = datetime.datetime.now().isoformat()
                self.event_writer.writerow([timestamp, frame_idx, event, ear, yaw, face_direction])
                self.event_log_handle.flush()
        except Exception as e:
            logging.error(f"Event logging error: {e}")
    
    def calculate_ear(self, landmarks, eye_indices):
        """Calculate Eye Aspect Ratio for given eye landmarks"""
        try:
            # Extract eye landmark coordinates
            points = []
            for idx in eye_indices:
                point = landmarks.landmark[idx]
                points.append((point.x, point.y))
            
            # DEBUG FIX 3: Ensure we have exactly 6 points for EAR calculation
            if len(points) != 6:
                logging.warning(f"Expected 6 eye points, got {len(points)}")
                return 0.0
            
            # Calculate vertical distances
            p2_p6 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))
            p3_p5 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))
            
            # Calculate horizontal distance
            p1_p4 = np.linalg.norm(np.array(points[0]) - np.array(points[3]))
            
            # EAR formula - add safety check for division by zero
            if p1_p4 == 0:
                return 0.0
                
            ear = (p2_p6 + p3_p5) / (2.0 * p1_p4)
            return ear
        except Exception as e:
            logging.error(f"EAR calculation error: {e}")
            return 0.0
    
    def get_eye_landmarks_indices(self):
        """Return indices for left and right eye landmarks in MediaPipe FaceMesh"""
        # DEBUG FIX 4: Correct eye landmark indices for MediaPipe
        left_eye = [33, 160, 158, 133, 153, 144]    # Left eye
        right_eye = [362, 385, 387, 263, 373, 380]  # Right eye
        return left_eye, right_eye
    
    def calculate_head_pose(self, landmarks, image_shape):
        """Calculate head pose using solvePnP"""
        try:
            # 3D model points for head pose estimation
            model_points = np.array([
                (0.0, 0.0, 0.0),        # Nose tip
                (0.0, -330.0, -65.0),   # Chin
                (-225.0, 170.0, -135.0),# Left eye left corner
                (225.0, 170.0, -135.0), # Right eye right corner
                (-150.0, -150.0, -125.0),# Left mouth corner
                (150.0, -150.0, -125.0) # Right mouth corner
            ], dtype=np.float64)
            
            # 2D image points from landmarks
            image_points = np.array([
                [landmarks.landmark[1].x * image_shape[1], landmarks.landmark[1].y * image_shape[0]],  # Nose tip
                [landmarks.landmark[152].x * image_shape[1], landmarks.landmark[152].y * image_shape[0]],  # Chin
                [landmarks.landmark[33].x * image_shape[1], landmarks.landmark[33].y * image_shape[0]],  # Left eye
                [landmarks.landmark[263].x * image_shape[1], landmarks.landmark[263].y * image_shape[0]],  # Right eye
                [landmarks.landmark[61].x * image_shape[1], landmarks.landmark[61].y * image_shape[0]],  # Left mouth
                [landmarks.landmark[291].x * image_shape[1], landmarks.landmark[291].y * image_shape[0]]   # Right mouth
            ], dtype=np.float64)
            
            # Camera matrix approximation
            focal_length = image_shape[1]
            center = (image_shape[1] / 2, image_shape[0] / 2)
            camera_matrix = np.array([
                [focal_length, 0, center[0]],
                [0, focal_length, center[1]],
                [0, 0, 1]
            ], dtype=np.float64)
            
            # Distortion coefficients (assuming no lens distortion)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points, image_points, camera_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                # Convert rotation vector to rotation matrix
                rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
                
                # Calculate Euler angles from rotation matrix
                euler_angles = self.rotation_matrix_to_euler_angles(rotation_matrix)
                yaw = np.degrees(euler_angles[1])  # Yaw angle in degrees
                
                return yaw, rotation_matrix, translation_vector
            
            return 0, None, None
        except Exception as e:
            logging.error(f"Head pose calculation error: {e}")
            return 0, None, None
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles (pitch, yaw, roll)"""
        try:
            sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
            
            singular = sy < 1e-6
            
            if not singular:
                x = np.arctan2(R[2, 1], R[2, 2])
                y = np.arctan2(-R[2, 0], sy)
                z = np.arctan2(R[1, 0], R[0, 0])
            else:
                x = np.arctan2(-R[1, 2], R[1, 1])
                y = np.arctan2(-R[2, 0], sy)
                z = 0
            
            return np.array([x, y, z])
        except Exception as e:
            logging.error(f"Euler angles conversion error: {e}")
            return np.array([0, 0, 0])
    
    def calculate_face_direction(self, pose_landmarks, image_shape):
        """Calculate face direction using pose landmarks (nose and ears)"""
        try:
            # Get key landmarks
            nose = pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # Extract coordinates
            nose_x = nose.x
            left_ear_x = left_ear.x
            right_ear_x = right_ear.x
            
            # Calculate ear midpoint
            ear_mid_x = (left_ear_x + right_ear_x) / 2
            
            # Calculate direction value
            direction_value = nose_x - ear_mid_x
            
            # Add to buffer for smoothing
            self.direction_buffer.append(direction_value)
            smoothed_value = np.mean(self.direction_buffer)
            
            # Calculate confidence based on landmark visibility
            confidence = min(nose.visibility, left_ear.visibility, right_ear.visibility)
            
            # Determine direction with adjusted thresholds
            if smoothed_value < self.LEFT_THRESHOLD:
                direction = "Left"
                direction_color = (0, 165, 255)  # Orange
            elif smoothed_value > self.RIGHT_THRESHOLD:
                direction = "Right" 
                direction_color = (0, 165, 255)  # Orange
            else:
                direction = "Forward"
                direction_color = (0, 255, 0)    # Green
            
            return direction, direction_value, confidence, direction_color, {
                'nose_x': nose_x,
                'left_ear_x': left_ear_x,
                'right_ear_x': right_ear_x,
                'ear_mid_x': ear_mid_x,
                'direction_value': direction_value,
                'smoothed_value': smoothed_value
            }
            
        except Exception as e:
            logging.error(f"Face direction calculation error: {e}")
            return "Unknown", 0, 0, (255, 255, 255), {}
    
    def visualize_pose_landmarks(self, image, pose_landmarks, face_direction, direction_value, details):
        """Visualize pose landmarks with beautiful styling"""
        h, w = image.shape[:2]
        
        try:
            # Draw pose landmarks with custom styling
            self.mp_drawing.draw_landmarks(
                image,
                pose_landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            
            # Convert normalized coordinates to pixel coordinates
            nose_x_px = int(details['nose_x'] * w)
            left_ear_x_px = int(details['left_ear_x'] * w)
            right_ear_x_px = int(details['right_ear_x'] * w)
            ear_mid_x_px = int(details['ear_mid_x'] * w)
            
            nose_y = int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE].y * h)
            ear_y = int(pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR].y * h)
            
            # Draw ear midpoint line (vertical) - Cyan
            cv2.line(image, (ear_mid_x_px, 0), (ear_mid_x_px, h), (255, 255, 0), 2)
            
            # Draw nose to ear midpoint line - Yellow
            cv2.line(image, (ear_mid_x_px, nose_y), (nose_x_px, nose_y), (0, 255, 255), 2)
            
            # Draw landmarks with beautiful colors and sizes
            cv2.circle(image, (nose_x_px, nose_y), 10, (0, 255, 255), -1)  # Nose - Yellow
            cv2.circle(image, (nose_x_px, nose_y), 10, (0, 0, 0), 2)       # Nose border
            
            cv2.circle(image, (left_ear_x_px, ear_y), 8, (255, 0, 0), -1)   # Left ear - Blue
            cv2.circle(image, (left_ear_x_px, ear_y), 8, (0, 0, 0), 2)      # Left ear border
            
            cv2.circle(image, (right_ear_x_px, ear_y), 8, (255, 0, 0), -1)  # Right ear - Blue
            cv2.circle(image, (right_ear_x_px, ear_y), 8, (0, 0, 0), 2)     # Right ear border
            
            cv2.circle(image, (ear_mid_x_px, ear_y), 6, (255, 255, 0), -1)  # Ear mid - Cyan
            cv2.circle(image, (ear_mid_x_px, ear_y), 6, (0, 0, 0), 2)       # Ear mid border
            
            # Draw direction indicator with animation
            indicator_x = w - 100
            indicator_y = 50
            
            if face_direction == "Left":
                cv2.arrowedLine(image, (indicator_x + 30, indicator_y), (indicator_x, indicator_y), 
                              (0, 165, 255), 10, tipLength=0.7)
            elif face_direction == "Right":
                cv2.arrowedLine(image, (indicator_x, indicator_y), (indicator_x + 30, indicator_y), 
                              (0, 165, 255), 10, tipLength=0.7)
            else:  # Forward
                cv2.circle(image, (indicator_x + 15, indicator_y), 20, (0, 255, 0), -1)
                cv2.circle(image, (indicator_x + 15, indicator_y), 20, (0, 0, 0), 2)
                
        except Exception as e:
            logging.error(f"Pose visualization error: {e}")
    
    def get_face_bounding_box(self, landmarks, image_shape):
        """Get bounding box for face from landmarks with generous padding for hand detection"""
        try:
            x_coords = [landmark.x for landmark in landmarks.landmark]
            y_coords = [landmark.y for landmark in landmarks.landmark]
            
            x_min = int(min(x_coords) * image_shape[1])
            y_min = int(min(y_coords) * image_shape[0])
            x_max = int(max(x_coords) * image_shape[1])
            y_max = int(max(y_coords) * image_shape[0])
            
            # Store original face box (for visualization if needed)
            self.original_face_box = (x_min, y_min, x_max, y_max)
            
            # FIX: Generous padding to detect hands near head area (ears, sides, etc.)
            padding_x = int((x_max - x_min) * 0.3)   # 80% horizontal padding 
            padding_y = int((y_max - y_min) * 0.1)   # 100% vertical padding
            
            # Apply padding
            x_min = max(0, x_min - padding_x)
            y_min = max(0, y_min - padding_y)
            x_max = min(image_shape[1], x_max + padding_x)
            y_max = min(image_shape[0], y_max + padding_y)
            
            return (x_min, y_min, x_max, y_max)
        except Exception as e:
            logging.error(f"Face bounding box error: {e}")
            return None
    
    def get_hand_bounding_boxes(self, hand_landmarks_list, image_shape):
        """Get bounding boxes for all detected hands"""
        hand_boxes = []
        try:
            for hand_landmarks in hand_landmarks_list:
                x_coords = [landmark.x for landmark in hand_landmarks.landmark]
                y_coords = [landmark.y for landmark in hand_landmarks.landmark]
                
                x_min = int(min(x_coords) * image_shape[1])
                y_min = int(min(y_coords) * image_shape[0])
                x_max = int(max(x_coords) * image_shape[1])
                y_max = int(max(y_coords) * image_shape[0])
                
                hand_boxes.append((x_min, y_min, x_max, y_max))
        except Exception as e:
            logging.error(f"Hand bounding box error: {e}")
        
        return hand_boxes
    
    def calculate_overlap_area(self, box1, box2):
        """Calculate overlap area between two bounding boxes"""
        try:
            x1_min, y1_min, x1_max, y1_max = box1
            x2_min, y2_min, x2_max, y2_max = box2
            
            # Calculate intersection
            x_overlap = max(0, min(x1_max, x2_max) - max(x1_min, x2_min))
            y_overlap = max(0, min(y1_max, y2_max) - max(y1_min, y2_min))
            
            overlap_area = x_overlap * y_overlap
            return overlap_area
        except Exception as e:
            logging.error(f"Overlap calculation error: {e}")
            return 0
    
    def calculate_box_area(self, box):
        """Calculate area of a bounding box"""
        try:
            x_min, y_min, x_max, y_max = box
            return (x_max - x_min) * (y_max - y_min)
        except Exception as e:
            logging.error(f"Box area calculation error: {e}")
            return 0
    
    def play_beep(self):
        """Play beep sound (Windows winsound or fallback)"""
        try:
            if HAS_WINSOUND:
                winsound.Beep(2500, 500)
            else:
                # Fallback for non-Windows systems
                print('\a')  # Terminal bell
        except Exception as e:
            logging.error(f"Beep error: {e}")
    
    def speak_alert(self, message):
        """Speak alert message in a non-blocking thread"""
        def speak():
            try:
                with self.tts_lock:
                    if self.tts_available and not self.is_speaking:
                        self.is_speaking = True
                        self.tts_engine.say(message)
                        self.tts_engine.runAndWait()
                        self.is_speaking = False
            except Exception as e:
                logging.error(f"TTS error: {e}")
                self.is_speaking = False
        
        if self.tts_available:
            thread = threading.Thread(target=speak)
            thread.daemon = True
            thread.start()
        else:
            print(f"ALERT: {message}")
    
    def update_display_info(self, image, ear, yaw, face_direction, state, looking_direction, hand_near_face, face_detected):
        """Update the display with current information"""
        try:
            # Define colors based on state
            if state == "Sleepy":
                color = (0, 0, 255)  # Red
            elif state in ["Distracted", "HandNear"]:
                color = (0, 165, 255)  # Orange
            elif state == "FaceNotDetected":
                color = (255, 0, 0)  # Bright Red for face detection failure
            else:
                color = (0, 255, 0)  # Green
            
            # Display state information
            cv2.putText(image, f"State: {state}", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Display face detection status
            if not face_detected:
                cv2.putText(image, "FACE NOT DETECTED!", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            else:
                cv2.putText(image, f"EAR: {ear:.3f}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Yaw: {yaw:.1f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Face: {face_direction}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(image, f"Looking: {looking_direction}", (10, 150), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display hand near face warning if applicable
            if hand_near_face:
                cv2.putText(image, "HAND NEAR FACE!", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        except Exception as e:
            logging.error(f"Display update error: {e}")
    
    def safe_state_change(self, new_state):
        """Safely change state with cooldown to prevent rapid flipping"""
        current_time = time.time()
        if current_time - self.last_state_change > self.state_cooldown:
            if self.current_state != new_state:
                self.current_state = new_state
                self.last_state_change = current_time
                return True
        return False
    
    def is_face_properly_detected(self, avg_ear, yaw_angle, face_results, pose_results):
        """Check if face is properly detected with valid metrics"""
        # Check if face landmarks are detected
        if not face_results.multi_face_landmarks:
            return False
        
        # Check if EAR and Yaw values are valid (not stuck at 0)
        if avg_ear < self.ZERO_EAR_THRESHOLD and abs(yaw_angle) < self.ZERO_YAW_THRESHOLD:
            return False
        
        # Check if pose landmarks are available for face direction
        if pose_results.pose_landmarks:
            nose = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.NOSE]
            left_ear = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.LEFT_EAR]
            right_ear = pose_results.pose_landmarks.landmark[self.mp_pose.PoseLandmark.RIGHT_EAR]
            
            # Check visibility of key landmarks
            if nose.visibility < 0.5 or left_ear.visibility < 0.3 or right_ear.visibility < 0.3:
                return False
        
        return True
    
    def run(self):
        """Main loop for the driver monitoring system"""
        frame_count = 0
        start_time = time.time()
        
        print("Starting Driver Monitoring System...")
        print("Press ESC to exit")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                if not ret:
                    logging.error("Failed to grab frame")
                    break
                
                frame_count += 1
                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image_shape = frame.shape
                
                # Process frame with MediaPipe
                face_results = self.face_mesh.process(image_rgb)
                hand_results = self.hands.process(image_rgb)
                pose_results = self.pose.process(image_rgb)
                
                # Initialize default values
                ear_left = ear_right = 0.0
                avg_ear = 0.0
                yaw_angle = 0.0
                face_direction = "Unknown"
                direction_value = 0.0
                face_direction_confidence = 0.0
                face_box = None
                hand_boxes = []
                pose_details = {}
                
                # Process pose landmarks for face direction
                if pose_results.pose_landmarks:
                    face_direction, direction_value, face_direction_confidence, _, pose_details = self.calculate_face_direction(
                        pose_results.pose_landmarks, image_shape
                    )
                    
                    # Only use pose-based direction if confidence is good
                    if face_direction_confidence > self.FACE_DIRECTION_CONFIDENCE_THRESHOLD:
                        self.face_direction = face_direction
                        # Visualize pose landmarks
                        self.visualize_pose_landmarks(frame, pose_results.pose_landmarks, 
                                                    face_direction, direction_value, pose_details)
                
                # Process face landmarks for EAR and head pose
                if face_results.multi_face_landmarks:
                    face_landmarks = face_results.multi_face_landmarks[0]
                    
                    # Calculate EAR for both eyes
                    left_eye_indices, right_eye_indices = self.get_eye_landmarks_indices()
                    ear_left = self.calculate_ear(face_landmarks, left_eye_indices)
                    ear_right = self.calculate_ear(face_landmarks, right_eye_indices)
                    avg_ear = (ear_left + ear_right) / 2.0
                    
                    # Calculate head pose
                    yaw_angle, _, _ = self.calculate_head_pose(face_landmarks, image_shape)
                    
                    # Smooth yaw angle
                    self.yaw_buffer.append(yaw_angle)
                    smoothed_yaw = np.mean(self.yaw_buffer)
                    
                    # Get face bounding box
                    face_box = self.get_face_bounding_box(face_landmarks, image_shape)
                    
                    # Draw face bounding box (padded boundary)
                    if face_box:
                        x_min, y_min, x_max, y_max = face_box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
                        
                        # Also draw original face box for reference (thin line)
                        if hasattr(self, 'original_face_box'):
                            orig_x_min, orig_y_min, orig_x_max, orig_y_max = self.original_face_box
                            cv2.rectangle(frame, (orig_x_min, orig_y_min), (orig_x_max, orig_y_max), (255, 255, 0), 1)
                
                # Process hand landmarks
                if hand_results.multi_hand_landmarks:
                    hand_boxes = self.get_hand_bounding_boxes(hand_results.multi_hand_landmarks, image_shape)
                    
                    # Draw hand landmarks and bounding boxes
                    for hand_landmarks, hand_box in zip(hand_results.multi_hand_landmarks, hand_boxes):
                        self.mp_drawing.draw_landmarks(
                            frame,
                            hand_landmarks,
                            self.mp_hands.HAND_CONNECTIONS
                        )
                        
                        # Draw hand bounding box
                        x_min, y_min, x_max, y_max = hand_box
                        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 0, 255), 2)
                
                current_time = time.time()
                
                # FACE DETECTION FAILURE DETECTION (Highest Priority)
                face_properly_detected = self.is_face_properly_detected(avg_ear, yaw_angle, face_results, pose_results)
                
                if not face_properly_detected:
                    if self.face_not_detected_start is None:
                        self.face_not_detected_start = current_time
                        self.log_event(frame_count, "FaceNotDetectedStarted", avg_ear, yaw_angle, face_direction)
                    
                    face_not_detected_duration = current_time - self.face_not_detected_start
                    
                    # Alert after 3 seconds of no face detection
                    if face_not_detected_duration > 3:
                        self.play_beep()
                    
                    if face_not_detected_duration > 5 and (current_time - self.last_vocal_alert) > self.vocal_alert_interval:
                        self.speak_alert("Unable to detect your face! Please make sure you are awake and positioned correctly in front of the camera.")
                        self.last_vocal_alert = current_time
                        self.log_event(frame_count, "FaceDetectionAlert", avg_ear, yaw_angle, face_direction)
                    
                    self.face_detected = False
                    new_state = "FaceNotDetected"
                else:
                    if self.face_not_detected_start is not None:
                        self.log_event(frame_count, "FaceDetected", avg_ear, yaw_angle, face_direction)
                    self.face_not_detected_start = None
                    self.face_detected = True
                    new_state = "Concentrated"
                
                # Only proceed with other detections if face is properly detected
                if face_properly_detected:
                    # DROWSINESS DETECTION (High Priority)
                    if avg_ear < self.EAR_THRESHOLD and avg_ear > 0:
                        if self.eyes_closed_start is None:
                            self.eyes_closed_start = current_time
                            self.log_event(frame_count, "EyesClosedStarted", avg_ear, yaw_angle, face_direction)
                        
                        eyes_closed_duration = current_time - self.eyes_closed_start
                        
                        if eyes_closed_duration > 2:
                            self.play_beep()
                        
                        if eyes_closed_duration > 4 and (current_time - self.last_vocal_alert) > self.vocal_alert_interval:
                            self.speak_alert("Wake up! Please pay attention.")
                            self.last_vocal_alert = current_time
                            self.log_event(frame_count, "DrowsinessAlert", avg_ear, yaw_angle, face_direction)
                        
                        new_state = "Sleepy"
                    else:
                        if self.eyes_closed_start is not None:
                            self.log_event(frame_count, "EyesClosedEnded", avg_ear, yaw_angle, face_direction)
                        self.eyes_closed_start = None
                    
                    # DISTRACTION DETECTION (Medium Priority)
                    if abs(yaw_angle) > self.YAW_THRESHOLD or face_direction in ["Left", "Right"]:
                        if self.distracted_start is None:
                            self.distracted_start = current_time
                            self.log_event(frame_count, "DistractionStarted", avg_ear, yaw_angle, face_direction)
                        
                        distracted_duration = current_time - self.distracted_start
                        
                        if distracted_duration > 2:
                            self.play_beep()
                        
                        if distracted_duration > 4 and (current_time - self.last_vocal_alert) > self.vocal_alert_interval:
                            self.speak_alert("Hey, you are distracted. Concentrate!")
                            self.last_vocal_alert = current_time
                            self.log_event(frame_count, "DistractionAlert", avg_ear, yaw_angle, face_direction)
                        
                        # Only set to Distracted if not already Sleepy
                        if new_state != "Sleepy":
                            new_state = "Distracted"
                        
                        # Determine looking direction
                        if yaw_angle > 0 or face_direction == "Right":
                            self.looking_direction = "Right"
                        elif yaw_angle < 0 or face_direction == "Left":
                            self.looking_direction = "Left"
                        else:
                            self.looking_direction = "Center"
                    else:
                        if self.distracted_start is not None:
                            self.log_event(frame_count, "DistractionEnded", avg_ear, yaw_angle, face_direction)
                        self.distracted_start = None
                        self.looking_direction = "Center"
                    
                    # HAND-NEAR-FACE DETECTION (Lowest Priority)
                    hand_near_detected = False
                    if face_box and hand_boxes:
                        face_x_min, face_y_min, face_x_max, face_y_max = face_box
                        
                        for hand_box in hand_boxes:
                            hand_x_min, hand_y_min, hand_x_max, hand_y_max = hand_box
                            
                            # Simple boundary intersection check
                            if (hand_x_min < face_x_max and hand_x_max > face_x_min and
                                hand_y_min < face_x_max and hand_y_max > face_y_min):
                                hand_near_detected = True
                                break

                    if hand_near_detected:
                        if not self.hand_near_face:
                            self.hand_near_start = current_time
                            self.hand_near_face = True
                            self.log_event(frame_count, "HandNearStarted", avg_ear, yaw_angle, face_direction)
                        
                        hand_near_duration = current_time - self.hand_near_start
                        
                        if hand_near_duration > 4 and (current_time - self.last_hand_alert) > self.vocal_alert_interval:
                            self.speak_alert("You are holding something. It may cause you distracted. Keep it away, stay focused.")
                            self.last_hand_alert = current_time
                            self.log_event(frame_count, "HandNearAlert", avg_ear, yaw_angle, face_direction)
                        
                        # Only set to HandNear if not already Sleepy or Distracted
                        if new_state not in ["Sleepy", "Distracted"]:
                            new_state = "HandNear"
                    else:
                        if self.hand_near_face:
                            # RESET the hand alert timer when hand near face state ends
                            self.last_hand_alert = 0  # ADD THIS LINE
                            self.log_event(frame_count, "HandNearEnded", avg_ear, yaw_angle, face_direction)
                        self.hand_near_face = False
                        self.hand_near_start = None
                
                # DEBUG FIX 7: Apply state change with cooldown
                self.safe_state_change(new_state)
                
                # Update display with current information
                self.update_display_info(frame, avg_ear, yaw_angle, self.face_direction, self.current_state, 
                                       self.looking_direction, self.hand_near_face, self.face_detected)
                
                # Log frame data
                self.log_frame(frame_count, avg_ear, yaw_angle, self.face_direction, self.current_state)
                
                # Calculate and display FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, image_shape[0] - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                # Display the frame
                cv2.imshow('Driver Monitoring System', frame)
                
                # Check for exit key
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key
                    break
                    
        except Exception as e:
            logging.error(f"Main loop error: {e}")
            import traceback
            traceback.print_exc()
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Clean up resources"""
        self.running = False
        print("Cleaning up...")
        
        # Close log files
        try:
            if self.frame_log_handle:
                self.frame_log_handle.close()
            if self.event_log_handle:
                self.event_log_handle.close()
        except Exception as e:
            logging.error(f"Log file close error: {e}")
        
        # Release camera
        try:
            self.cap.release()
        except Exception as e:
            logging.error(f"Camera release error: {e}")
        
        # Close windows
        cv2.destroyAllWindows()
        
        print("Cleanup completed!")

if __name__ == "__main__":
    try:
        system = DriverMonitoringSystem()
        system.run()
    except Exception as e:
        logging.error(f"System initialization error: {e}")
        import traceback
        traceback.print_exc()