import cv2
import mediapipe as mp
import os
import time
import numpy as np

# ─────────────────────────────────────────────
#  CONFIG
# ─────────────────────────────────────────────
BASE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dataset")

CLASSES = {
    "1": "alert",
    "2": "microsleep",
    "3": "drowsy",
    "4": "extreme_fatigue",
}

TARGET_PER_CLASS = 500
CAPTURE_INTERVAL = 0.1
IMG_SIZE = (224, 224)

COLOR_ALERT      = (50,  205,  50)
COLOR_MICROSLEEP = (0,   200, 255)
COLOR_DROWSY     = (0,   140, 255)
COLOR_EXTREME    = (0,     0, 220)
COLOR_INACTIVE   = (120, 120, 120)

CLASS_COLORS = {
    "alert":           COLOR_ALERT,
    "microsleep":      COLOR_MICROSLEEP,
    "drowsy":          COLOR_DROWSY,
    "extreme_fatigue": COLOR_EXTREME,
}

# ─────────────────────────────────────────────
#  NEW MEDIAPIPE 0.10.x API SETUP
# ─────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_landmarker.task")

if not os.path.exists(MODEL_PATH):
    print("Downloading MediaPipe face landmarker model (~6 MB)...")
    import urllib.request
    url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
    urllib.request.urlretrieve(url, MODEL_PATH)
    print("Model downloaded successfully!")

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

# Landmark connections for manual drawing
EYE_CONNECTIONS = [
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133),
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),(249,263),
    (362,398),(398,384),(384,385),(385,386),(386,387),(387,388),(388,466),(466,263),
]
MOUTH_CONNECTIONS = [
    (61,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291),
    (291,409),(409,270),(270,269),(269,267),(267,0),(0,37),(37,39),(39,40),(40,185),(185,61),
]
FACE_OVAL = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
]

# ─────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────
def count_existing(class_name):
    folder = os.path.join(BASE_DIR, class_name)
    os.makedirs(folder, exist_ok=True)
    return len([f for f in os.listdir(folder) if f.endswith(".jpg")])


def next_filename(class_name):
    folder = os.path.join(BASE_DIR, class_name)
    existing = count_existing(class_name)
    return os.path.join(folder, f"{class_name}_{existing:05d}.jpg")


def draw_landmarks_manual(frame, landmarks, h, w):
    def pt(idx):
        lm = landmarks[idx]
        return (int(lm.x * w), int(lm.y * h))
    for a, b in FACE_OVAL:
        cv2.line(frame, pt(a), pt(b), (80, 80, 80), 1, cv2.LINE_AA)
    for a, b in EYE_CONNECTIONS:
        cv2.line(frame, pt(a), pt(b), (0, 220, 180), 1, cv2.LINE_AA)
    for a, b in MOUTH_CONNECTIONS:
        cv2.line(frame, pt(a), pt(b), (180, 100, 220), 1, cv2.LINE_AA)


def draw_ui(frame, active_class, counts, face_detected, capturing, countdown):
    h, w = frame.shape[:2]

    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (260, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    cv2.putText(frame, "DROWSINESS DATASET", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.putText(frame, "COLLECTOR", (10, 52),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 220, 220), 1, cv2.LINE_AA)
    cv2.line(frame, (10, 60), (250, 60), (80, 80, 80), 1)

    y = 85
    for key, name in CLASSES.items():
        count     = counts[name]
        pct       = min(count / TARGET_PER_CLASS, 1.0)
        is_active = (name == active_class and capturing)
        color     = CLASS_COLORS[name] if is_active else COLOR_INACTIVE

        cv2.putText(frame, f"[{key}] {name.replace('_',' ').upper()}", (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"{count}/{TARGET_PER_CLASS}", (185, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                    (200,200,200) if is_active else (100,100,100), 1, cv2.LINE_AA)

        bx, by, bw, bh = 10, y+5, 240, 5
        cv2.rectangle(frame, (bx, by), (bx+bw, by+bh), (60,60,60), -1)
        cv2.rectangle(frame, (bx, by), (bx+int(bw*pct), by+bh), color, -1)
        y += 52

    cv2.line(frame, (10, y), (250, y), (80, 80, 80), 1)
    for inst in ["SPACE  start/stop", "1-4    switch class", "Q      quit"]:
        y += 18
        cv2.putText(frame, inst, (10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (160,160,160), 1, cv2.LINE_AA)

    face_color = (50,205,50) if face_detected else (0,0,220)
    cv2.putText(frame, "FACE DETECTED" if face_detected else "NO FACE DETECTED",
                (10, h-50), cv2.FONT_HERSHEY_SIMPLEX, 0.40, face_color, 1, cv2.LINE_AA)

    if capturing and active_class:
        color = CLASS_COLORS.get(active_class, COLOR_INACTIVE)
        cv2.putText(frame, f"REC  {active_class.upper()}", (10, h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (262, 0), (w-1, h-1), color, 3)
    else:
        cv2.putText(frame, "PAUSED  (SPACE to start)", (10, h-25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120,120,120), 1, cv2.LINE_AA)

    if countdown > 0:
        cv2.putText(frame, str(countdown), (w//2-20, h//2+20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, (0,255,255), 4, cv2.LINE_AA)
    return frame


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    active_class    = "alert"
    capturing       = False
    last_capture    = 0.0
    countdown       = 0
    countdown_start = 0.0

    counts = {name: count_existing(name) for name in CLASSES.values()}

    print("\n" + "="*50)
    print("  DROWSINESS DATASET COLLECTOR  (MediaPipe 0.10.x)")
    print("="*50)
    print("  1-4   select class")
    print("  SPACE start / stop capture")
    print("  Q     quit")
    print("="*50 + "\n")

    with FaceLandmarker.create_from_options(options) as landmarker:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("ERROR: Failed to read frame.")
                break

            frame = cv2.flip(frame, 1)
            h, w  = frame.shape[:2]

            mp_image         = mp.Image(image_format=mp.ImageFormat.SRGB,
                                        data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            detection_result = landmarker.detect(mp_image)
            face_detected    = len(detection_result.face_landmarks) > 0

            if face_detected:
                draw_landmarks_manual(frame, detection_result.face_landmarks[0], h, w)

            now = time.time()

            if countdown > 0:
                remaining = 3 - int(now - countdown_start)
                if remaining > 0:
                    countdown = remaining
                else:
                    countdown    = 0
                    capturing    = True
                    last_capture = now

            if capturing and face_detected and active_class and countdown == 0:
                if now - last_capture >= CAPTURE_INTERVAL:
                    if counts[active_class] < TARGET_PER_CLASS:
                        save_frame = cv2.resize(frame[:, 260:], IMG_SIZE)
                        cv2.imwrite(next_filename(active_class), save_frame)
                        counts[active_class] += 1
                        last_capture = now
                    else:
                        capturing = False
                        print(f"✓ {active_class}: target of {TARGET_PER_CLASS} reached!")

            frame = draw_ui(frame, active_class, counts, face_detected, capturing, countdown)
            cv2.imshow("Drowsiness Dataset Collector", frame)

            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), ord("Q")):
                break
            elif key == ord(" "):
                if capturing:
                    capturing = False
                    countdown = 0
                    print(f"Paused  |  {active_class}: {counts[active_class]} images")
                else:
                    if face_detected:
                        countdown       = 3
                        countdown_start = now
                        capturing       = False
                        print(f"Starting in 3s...  class = {active_class}")
                    else:
                        print("No face detected — position yourself in the camera first")
            elif key in [ord(k) for k in CLASSES]:
                active_class = CLASSES[chr(key)]
                capturing    = False
                countdown    = 0
                print(f"Switched to: {active_class}  ({counts[active_class]} images so far)")

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print("  FINAL COUNTS")
    print("="*50)
    for name in CLASSES.values():
        fill = int((counts[name] / TARGET_PER_CLASS) * 20)
        print(f"  {name:<20} [{'█'*fill}{'░'*(20-fill)}]  {counts[name]}/{TARGET_PER_CLASS}")
    print("="*50 + "\n")


if __name__ == "__main__":
    main()