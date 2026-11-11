import cv2
import cvzone
from cvzone.FaceMeshModule import FaceMeshDetector
from ultralytics import YOLO
import pygame
import os
import time
import math
import threading

# ==============================================================================
# 1. AUDIO SUBSYSTEM INITIALIZATION
# ==============================================================================
pygame.mixer.init()

current_dir = os.path.dirname(os.path.abspath(__file__))
alarm_path = os.path.join(current_dir, "alarm.mp3")
faudio_path = os.path.join(current_dir, "faudio.mp3")
paudio_path = os.path.join(current_dir, "paudio.mp3")

alarm_sound = pygame.mixer.Sound(alarm_path)
faudio_sound = pygame.mixer.Sound(faudio_path)
paudio_sound = pygame.mixer.Sound(paudio_path)

# Dedicated audio channels to eliminate audio stutter and channel conflicts
alarm_channel = pygame.mixer.Channel(0)
face_channel = pygame.mixer.Channel(1)
phone_channel = pygame.mixer.Channel(2)

# ==============================================================================
# 2. TUNABLE DETECTION THRESHOLDS & STABILITY PARAMETERS
# ==============================================================================
# Eye Aspect Ratio (EAR) Thresholds:
# Open eyes typically ~25-35. Closed eyes drop to ~5-12.
EAR_CLOSE_THRESHOLD = 15.0         # Below this, eye is considered closed
EAR_RECOVERY_THRESHOLD = 19.0      # Hysteresis: must rise above this to recover
DROWSY_FRAMES_TRIGGER = 22         # Consecutive frames (~0.75s) to trigger sleep alarm (filters natural blinks)

# Face Detection Debounce:
NO_FACE_FRAMES_TRIGGER = 24        # Consecutive frames (~0.8s) before "Don't cover your face" alert

# Phone Detection (YOLO):
PHONE_CONF_THRESHOLD = 0.32        # Confidence threshold for cell phone
PHONE_PERSISTENCE_TIME = 0.45      # Seconds to keep detection active to prevent box blinking/flicker

# ==============================================================================
# 3. ASYNCHRONOUS YOLO WORKER THREAD (ZERO CAMERA LAG)
# ==============================================================================
class AsyncPhoneDetector(threading.Thread):
    """
    Runs YOLO phone detection asynchronously in a background thread.
    This guarantees the main video capture and FaceMesh run at a silky smooth
    30 FPS without any camera stutter or lag.
    """
    def __init__(self, model_path):
        super().__init__(daemon=True)
        self.model = YOLO(model_path)
        self.lock = threading.Lock()
        self.frame_to_process = None
        self.running = True
        
        # Shared detection state
        self.detected = False
        self.box = None
        self.confidence = 0.0
        self.last_detection_time = 0.0

    def submit_frame(self, frame):
        with self.lock:
            # Keep only the latest frame
            self.frame_to_process = frame.copy()

    def get_latest_detection(self):
        with self.lock:
            now = time.time()
            is_active = self.detected or (now - self.last_detection_time < PHONE_PERSISTENCE_TIME)
            return is_active, self.box, self.confidence

    def run(self):
        while self.running:
            img = None
            with self.lock:
                if self.frame_to_process is not None:
                    img = self.frame_to_process
                    self.frame_to_process = None

            if img is not None:
                # Run YOLO at 320x320 optimized specifically for class 67 (cell phone)
                results = self.model(img, imgsz=320, classes=[67], verbose=False)
                best_box = None
                best_conf = 0.0

                for r in results:
                    for b in r.boxes:
                        conf = float(b.conf[0])
                        if conf >= PHONE_CONF_THRESHOLD and conf > best_conf:
                            best_conf = conf
                            best_box = list(map(int, b.xyxy[0]))

                with self.lock:
                    if best_box is not None:
                        self.detected = True
                        self.box = best_box
                        self.confidence = best_conf
                        self.last_detection_time = time.time()
                    else:
                        self.detected = False
            else:
                time.sleep(0.01)

# ==============================================================================
# 4. MATH & BIOMETRIC RATIO HELPERS
# ==============================================================================
def distance(p1, p2):
    """Euclidean distance between two 2D points."""
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_robust_ear(face):
    """
    Calculates 6-point biometric Eye Aspect Ratio (EAR) averaged across both eyes.
    Using multiple vertical landmark pairs per eye drastically reduces noise from
    head pitch, yaw, and minor perspective distortion.
    """
    # --- Left Eye Landmarks ---
    # Horizontal: 33 (outer), 133 (inner)
    # Vertical pairs: (160, 144) and (158, 153)
    d_h_left = distance(face[33], face[133])
    d_v1_left = distance(face[160], face[144])
    d_v2_left = distance(face[158], face[153])
    ear_left = (d_v1_left + d_v2_left) / (2.0 * max(d_h_left, 1.0))

    # --- Right Eye Landmarks ---
    # Horizontal: 362 (inner), 263 (outer)
    # Vertical pairs: (385, 380) and (387, 373)
    d_h_right = distance(face[362], face[263])
    d_v1_right = distance(face[385], face[380])
    d_v2_right = distance(face[387], face[373])
    ear_right = (d_v1_right + d_v2_right) / (2.0 * max(d_h_right, 1.0))

    avg_ear = ((ear_left + ear_right) / 2.0) * 100.0
    return avg_ear

# ==============================================================================
# 5. UI DRAWING HELPERS (IDENTICAL TO VIDEO VISUALS)
# ==============================================================================
def draw_top_badge(img, text, bg_color=(255, 0, 255)):
    """Draws the top-left magenta badge (e.g. Eye Ratio: 25)."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.65
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    pad_x = 10
    pad_y = 8
    x1, y1 = 30, 25
    x2, y2 = x1 + text_w + pad_x * 2, y1 + text_h + pad_y * 2
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
    cv2.putText(img, text, (x1 + pad_x, y1 + pad_y + text_h - 2),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

def draw_alert_banner(img, text, bg_color):
    """Draws a solid filled alert banner centered near the top."""
    h, w, _ = img.shape
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = 0.85
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
    
    pad_x = 18
    pad_y = 12
    top_y = 70
    center_x = w // 2
    x1 = max(10, center_x - (text_w // 2) - pad_x)
    y1 = top_y
    x2 = min(w - 10, center_x + (text_w // 2) + pad_x)
    y2 = top_y + text_h + pad_y * 2
    
    cv2.rectangle(img, (x1, y1), (x2, y2), bg_color, cv2.FILLED)
    cv2.putText(img, text, (x1 + pad_x, y1 + pad_y + text_h),
                font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)

# ==============================================================================
# 6. MAIN APPLICATION EXECUTION
# ==============================================================================
def main():
    print("=" * 60)
    print("  Smart Study Monitor [Enhanced Edition]")
    print("  - DirectShow Zero-Latency Video Backend")
    print("  - Asynchronous High-Speed YOLO Phone Detection")
    print("  - 6-Point Biometric Eye Ratio with EMA Noise Filter")
    print("  - Natural Blink Filtering & State Hysteresis")
    print("  Press 'q' or 'ESC' to exit.")
    print("=" * 60)

    # Open camera with DirectShow backend on Windows (avoids MSMF freezing and latency)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    if not cap.isOpened():
        print("Notice: DirectShow camera unavailable, falling back to default backend...")
        cap = cv2.VideoCapture(0)

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1) # Prevent frame queue latency

    # Initialize FaceMesh Detector
    detector = FaceMeshDetector(maxFaces=1)

    # Initialize Async Phone Detector Thread
    model_path = os.path.join(current_dir, "yolov8n.pt")
    phone_thread = AsyncPhoneDetector(model_path)
    phone_thread.start()

    # State tracking variables
    smoothed_ear = 28.0            # Exponential Moving Average value for Eye Ratio
    drowsy_counter = 0
    drowsy_alert_active = False
    no_face_counter = 0
    
    # Smooth bounding box interpolation (prevents jittery box borders)
    smooth_box = None

    frame_count = 0

    try:
        while True:
            success, img = cap.read()
            if not success:
                print("Warning: Camera frame read failed. Retrying...")
                time.sleep(0.05)
                continue

            frame_count += 1

            # Submit frame to background phone detector every 2nd frame for optimal throughput
            if frame_count % 2 == 0:
                phone_thread.submit_frame(img)

            # 1. Real-time Face Mesh Detection
            img, faces = detector.findFaceMesh(img, draw=False)

            # 2. Check Async Phone Detection Status
            phone_active, target_box, phone_conf = phone_thread.get_latest_detection()

            if phone_active and target_box is not None:
                # Smooth box interpolation using Linear Interpolation (LERP)
                if smooth_box is None:
                    smooth_box = [float(c) for c in target_box]
                else:
                    lerp_rate = 0.4
                    smooth_box[0] += (target_box[0] - smooth_box[0]) * lerp_rate
                    smooth_box[1] += (target_box[1] - smooth_box[1]) * lerp_rate
                    smooth_box[2] += (target_box[2] - smooth_box[2]) * lerp_rate
                    smooth_box[3] += (target_box[3] - smooth_box[3]) * lerp_rate

                bx1, by1, bx2, by2 = map(int, smooth_box)

                # Draw magenta bounding box
                cv2.rectangle(img, (bx1, by1), (bx2, by2), (255, 0, 255), 3)

                # Draw magenta confidence label badge
                label = f"Phone detected! {int(phone_conf * 100)}%"
                font = cv2.FONT_HERSHEY_SIMPLEX
                (lw, lh), _ = cv2.getTextSize(label, font, 0.55, 2)
                badge_y1 = max(0, by1 - lh - 12)
                badge_y2 = by1
                cv2.rectangle(img, (bx1, badge_y1), (bx1 + lw + 14, badge_y2), (255, 0, 255), cv2.FILLED)
                cv2.putText(img, label, (bx1 + 7, badge_y2 - 5), font, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                smooth_box = None

            # 3. State & Behavioral Evaluation
            if phone_active:
                # Priority 1: Phone distraction detected
                draw_alert_banner(img, "PUT THE PHONE AWAY!", bg_color=(0, 165, 255)) # Orange
                if not phone_channel.get_busy():
                    phone_channel.play(paudio_sound)
                alarm_channel.stop()
                face_channel.stop()
                drowsy_alert_active = False

            elif not faces:
                # Priority 2: Face not detected / face covered by book or turned away
                phone_channel.stop()
                alarm_channel.stop()
                drowsy_alert_active = False
                drowsy_counter = 0

                no_face_counter += 1
                if no_face_counter >= NO_FACE_FRAMES_TRIGGER:
                    draw_alert_banner(img, "DONT COVER YOUR FACE!", bg_color=(0, 0, 230)) # Red
                    if not face_channel.get_busy():
                        face_channel.play(faudio_sound)
                else:
                    face_channel.stop()

            else:
                # Priority 3: Face visible -> Check Eye Ratio for Drowsiness
                phone_channel.stop()
                face_channel.stop()
                no_face_counter = 0

                face = faces[0]
                raw_ear = calculate_robust_ear(face)

                # Exponential Moving Average (EMA) to smooth out micro-twitches and sensor noise
                alpha = 0.35
                smoothed_ear = (alpha * raw_ear) + ((1.0 - alpha) * smoothed_ear)
                display_ratio = int(round(smoothed_ear))

                # Display Eye Ratio magenta badge at top-left
                draw_top_badge(img, f"Eye Ratio: {display_ratio}", bg_color=(255, 0, 255))

                # Drowsiness detection with hysteresis:
                # - Triggers when smoothed_ear is below EAR_CLOSE_THRESHOLD for consecutive frames
                # - Recovers only when smoothed_ear rises above EAR_RECOVERY_THRESHOLD
                if smoothed_ear < EAR_CLOSE_THRESHOLD:
                    drowsy_counter += 1
                    if drowsy_counter >= DROWSY_FRAMES_TRIGGER:
                        drowsy_alert_active = True
                else:
                    if smoothed_ear > EAR_RECOVERY_THRESHOLD:
                        drowsy_counter = max(0, drowsy_counter - 2)
                        if drowsy_counter == 0:
                            drowsy_alert_active = False

                if drowsy_alert_active:
                    draw_alert_banner(img, "WAKE UP & STUDY!", bg_color=(0, 0, 230)) # Red
                    if not alarm_channel.get_busy():
                        alarm_channel.play(alarm_sound)
                else:
                    alarm_channel.stop()

            # Render display window matching the video title
            cv2.imshow("Smart Study Monitor", img)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q') or key == 27: # 'q' or ESC to quit
                break

    finally:
        # Graceful cleanup
        phone_thread.running = False
        cap.release()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        phone_thread.join(timeout=1.0)
        print("Smart Study Monitor closed cleanly.")

if __name__ == "__main__":
    main()
