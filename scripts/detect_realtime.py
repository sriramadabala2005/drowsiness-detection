import matplotlib
matplotlib.use("Agg")

import cv2
import mediapipe as mp
import numpy as np
import joblib
import os
import time
import threading
from collections import deque

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "drowsiness_model.pkl")
MP_MODEL   = os.path.join(BASE_DIR, "scripts", "face_landmarker.task")

# ─────────────────────────────────────────────
#  ALERT SOUND (beep via winsound or fallback)
# ─────────────────────────────────────────────
try:
    import winsound
    def play_alert(level):
        if level == 1:
            winsound.Beep(800, 200)
        elif level == 2:
            winsound.Beep(600, 400)
        elif level == 3:
            for _ in range(3):
                winsound.Beep(500, 300)
                time.sleep(0.1)
except ImportError:
    def play_alert(level):
        print(f"\a")  # terminal bell fallback

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

mp_options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MP_MODEL),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)

# ─────────────────────────────────────────────
#  LANDMARK INDICES
# ─────────────────────────────────────────────
LEFT_EYE       = [362, 385, 387, 263, 373, 380]
RIGHT_EYE      = [33,  160, 158, 133, 153, 144]
MOUTH          = [61, 291, 39, 181, 0, 17, 269, 405]
HEAD_POSE_PTS  = [1, 152, 226, 446, 57, 287]

EYE_CONN = [
    (33,7),(7,163),(163,144),(144,145),(145,153),(153,154),(154,155),(155,133),
    (33,246),(246,161),(161,160),(160,159),(159,158),(158,157),(157,173),(173,133),
    (362,382),(382,381),(381,380),(380,374),(374,373),(373,390),(390,249),(249,263),
    (362,398),(398,384),(384,385),(385,386),(386,387),(387,388),(388,466),(466,263),
]
MOUTH_CONN = [
    (61,84),(84,17),(17,314),(314,405),(405,321),(321,375),(375,291),
    (291,409),(409,270),(270,269),(269,267),(267,0),(0,37),(37,39),(39,40),(40,185),(185,61),
]
OVAL_CONN = [
    (10,338),(338,297),(297,332),(332,284),(284,251),(251,389),(389,356),
    (356,454),(454,323),(323,361),(361,288),(288,397),(397,365),(365,379),
    (379,378),(378,400),(400,377),(377,152),(152,148),(148,176),(176,149),
    (149,150),(150,136),(136,172),(172,58),(58,132),(132,93),(93,234),
    (234,127),(127,162),(162,21),(21,54),(54,103),(103,67),(67,109),(109,10),
]

# ─────────────────────────────────────────────
#  DROWSINESS LEVELS
# ─────────────────────────────────────────────
LEVELS = {
    0: {
        "name":    "ALERT",
        "color":   (50, 205, 50),
        "bg":      (20, 60, 20),
        "alert":   False,
        "message": "Driver is alert and focused",
    },
    1: {
        "name":    "MICROSLEEP",
        "color":   (0, 200, 255),
        "bg":      (0, 40, 60),
        "alert":   True,
        "message": "Warning: Eyes closing slowly!",
    },
    2: {
        "name":    "DROWSY",
        "color":   (0, 140, 255),
        "bg":      (0, 30, 60),
        "alert":   True,
        "message": "Danger: Take a break now!",
    },
    3: {
        "name":    "EXTREME FATIGUE",
        "color":   (0, 0, 220),
        "bg":      (40, 0, 60),
        "alert":   True,
        "message": "CRITICAL: Pull over immediately!",
    },
}

CLASS_TO_LEVEL = {0: 0, 1: 1, 2: 2, 3: 3}

# ─────────────────────────────────────────────
#  FEATURE EXTRACTION
# ─────────────────────────────────────────────
def lm_coords(landmarks, indices, w, h):
    return np.array([(landmarks[i].x * w, landmarks[i].y * h)
                     for i in indices], dtype=np.float64)

def ear(coords):
    p1,p2,p3,p4,p5,p6 = coords
    return (np.linalg.norm(p2-p6) + np.linalg.norm(p3-p5)) / (2.0 * np.linalg.norm(p1-p4) + 1e-6)

def mar(coords):
    p1,p2,p3,p4,p5,p6,p7,p8 = coords
    v = (np.linalg.norm(p3-p7) + np.linalg.norm(p4-p8) + np.linalg.norm(p5-p6))
    return v / (3.0 * np.linalg.norm(p1-p2) + 1e-6)

def head_pose(landmarks, w, h):
    model_pts = np.array([
        [0.0,0.0,0.0],[0.0,-63.6,-12.5],[-43.3,32.7,-26.0],
        [43.3,32.7,-26.0],[-28.9,-28.9,-24.1],[28.9,-28.9,-24.1]
    ], dtype=np.float64)
    img_pts = lm_coords(landmarks, HEAD_POSE_PTS, w, h)
    cam     = np.array([[w,0,w/2],[0,w,h/2],[0,0,1]], dtype=np.float64)
    ok, rvec, tvec = cv2.solvePnP(model_pts, img_pts, cam,
                                   np.zeros((4,1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return 0.0, 0.0, 0.0
    rmat, _ = cv2.Rodrigues(rvec)
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(np.hstack((rmat, tvec)))
    return float(euler[0]), float(euler[1]), float(euler[2])

def extract_features(landmarks, w, h):
    le   = lm_coords(landmarks, LEFT_EYE,  w, h)
    re   = lm_coords(landmarks, RIGHT_EYE, w, h)
    mo   = lm_coords(landmarks, MOUTH,     w, h)
    lear = ear(le)
    rear = ear(re)
    aear = (lear + rear) / 2.0
    mval = mar(mo)
    pitch, yaw, roll = head_pose(landmarks, w, h)
    closed = 1 if aear < 0.20 else 0
    return [lear, rear, aear, mval, pitch, yaw, roll, closed], aear, mval

# ─────────────────────────────────────────────
#  DRAW HELPERS
# ─────────────────────────────────────────────
def draw_landmarks(frame, landmarks, h, w, color):
    def pt(i):
        lm = landmarks[i]
        return (int(lm.x * w), int(lm.y * h))
    for a, b in OVAL_CONN:
        cv2.line(frame, pt(a), pt(b), (60,60,60), 1, cv2.LINE_AA)
    for a, b in EYE_CONN:
        cv2.line(frame, pt(a), pt(b), color, 1, cv2.LINE_AA)
    for a, b in MOUTH_CONN:
        cv2.line(frame, pt(a), pt(b), (180,100,220), 1, cv2.LINE_AA)


def draw_rounded_rect(img, x1, y1, x2, y2, color, radius=12, thickness=-1):
    cv2.rectangle(img, (x1+radius, y1), (x2-radius, y2), color, thickness)
    cv2.rectangle(img, (x1, y1+radius), (x2, y2-radius), color, thickness)
    cv2.circle(img, (x1+radius, y1+radius), radius, color, thickness)
    cv2.circle(img, (x2-radius, y1+radius), radius, color, thickness)
    cv2.circle(img, (x1+radius, y2-radius), radius, color, thickness)
    cv2.circle(img, (x2-radius, y2-radius), radius, color, thickness)


def draw_bar(frame, x, y, w, h, value, max_val, color, label):
    pct = min(value / max_val, 1.0)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (50,50,50), -1)
    cv2.rectangle(frame, (x, y), (x+int(w*pct), y+h), color, -1)
    cv2.rectangle(frame, (x, y), (x+w, y+h), (80,80,80), 1)
    cv2.putText(frame, label, (x, y-6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"{value:.3f}", (x+w+6, y+h-2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (220,220,220), 1, cv2.LINE_AA)


def format_time(seconds):
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"

# ─────────────────────────────────────────────
#  MAIN DETECTION LOOP
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  REAL-TIME DROWSINESS DETECTION")
    print("="*55)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model not found at {MODEL_PATH}")
        print("Run train_model.py first.")
        return

    if not os.path.exists(MP_MODEL):
        print(f"ERROR: face_landmarker.task not found at {MP_MODEL}")
        return

    model = joblib.load(MODEL_PATH)
    print("  Model loaded successfully!")
    print("  Press Q to quit | Press R to reset stats\n")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("ERROR: Cannot open webcam.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    # ── state ──
    WINDOW_SIZE     = 15
    pred_buffer     = deque(maxlen=WINDOW_SIZE)
    ear_history     = deque(maxlen=60)
    current_level   = 0
    last_alert_time = 0
    alert_interval  = {0: 999, 1: 8, 2: 5, 3: 2}

    session_start   = time.time()
    level_times     = [0.0, 0.0, 0.0, 0.0]
    level_start     = time.time()
    prev_level      = 0

    fps_counter     = deque(maxlen=30)
    frame_count     = 0

    # alert thread flag
    alert_playing   = [False]

    def trigger_alert(level):
        if not alert_playing[0]:
            alert_playing[0] = True
            def _play():
                play_alert(level)
                alert_playing[0] = False
            threading.Thread(target=_play, daemon=True).start()

    with FaceLandmarker.create_from_options(mp_options) as landmarker:
        while True:
            t_start = time.time()
            ret, raw = cap.read()
            if not ret:
                break

            frame = cv2.flip(raw, 1)
            h, w  = frame.shape[:2]

            # ── create UI canvas (wider than webcam) ──
            PANEL_W = 340
            canvas  = np.zeros((h, w + PANEL_W, 3), dtype=np.uint8)
            canvas[:, :w] = frame

            # ── face detection ──
            mp_img   = mp.Image(image_format=mp.ImageFormat.SRGB,
                                data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            result   = landmarker.detect(mp_img)
            detected = len(result.face_landmarks) > 0

            avg_ear_val = 0.0
            mar_val     = 0.0

            if detected:
                landmarks = result.face_landmarks[0]
                feats, avg_ear_val, mar_val = extract_features(landmarks, w, h)
                ear_history.append(avg_ear_val)

                pred = model.predict([feats])[0]
                pred_buffer.append(pred)

                # smoothed prediction — majority vote
                if len(pred_buffer) >= 5:
                    counts  = np.bincount(list(pred_buffer), minlength=4)
                    smooth  = int(np.argmax(counts))
                    current_level = CLASS_TO_LEVEL[smooth]

                lv = LEVELS[current_level]
                draw_landmarks(canvas[:, :w], landmarks, h, w, lv["color"])

                # alert logic
                now = time.time()
                if current_level > 0 and (now - last_alert_time) > alert_interval[current_level]:
                    trigger_alert(current_level)
                    last_alert_time = now
            else:
                pred_buffer.clear()

            # ── update session time per level ──
            now = time.time()
            level_times[prev_level] += now - level_start
            level_start = now
            prev_level  = current_level

            # ── FPS ──
            fps_counter.append(time.time() - t_start)
            fps = 1.0 / (np.mean(fps_counter) + 1e-6)

            # ════════════════════════════════════════
            #  DRAW LEFT PANEL (webcam area overlays)
            # ════════════════════════════════════════
            lv    = LEVELS[current_level]
            color = lv["color"]
            bg    = lv["bg"]

            # colored border around webcam
            border_thick = 4 if current_level > 0 else 2
            cv2.rectangle(canvas, (0,0), (w-1, h-1), color, border_thick)

            # top status bar
            overlay = canvas.copy()
            cv2.rectangle(overlay, (0,0), (w, 55), (10,10,10), -1)
            cv2.addWeighted(overlay, 0.7, canvas, 0.3, 0, canvas)

            cv2.putText(canvas, "DROWSINESS DETECTION SYSTEM", (12, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200,200,200), 1, cv2.LINE_AA)
            cv2.putText(canvas, f"FPS: {fps:.0f}", (12, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120,120,120), 1, cv2.LINE_AA)

            session_elapsed = now - session_start
            cv2.putText(canvas, f"Session: {format_time(session_elapsed)}", (w-160, 44),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (120,120,120), 1, cv2.LINE_AA)

            # face detected indicator
            fd_color = (50,205,50) if detected else (0,0,200)
            fd_text  = "FACE DETECTED" if detected else "NO FACE DETECTED"
            cv2.circle(canvas, (w-20, 22), 7, fd_color, -1)
            cv2.putText(canvas, fd_text, (w-160, 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, fd_color, 1, cv2.LINE_AA)

            # ── bottom level badge ──
            badge_h = 80
            overlay2 = canvas.copy()
            cv2.rectangle(overlay2, (0, h-badge_h), (w, h), bg, -1)
            cv2.addWeighted(overlay2, 0.75, canvas, 0.25, 0, canvas)

            cv2.putText(canvas, lv["name"], (20, h-badge_h+38),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2, cv2.LINE_AA)
            cv2.putText(canvas, lv["message"], (20, h-badge_h+62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200,200,200), 1, cv2.LINE_AA)

            # level dots
            dot_x = w - 160
            for i in range(4):
                dc = LEVELS[i]["color"] if i <= current_level else (60,60,60)
                cv2.circle(canvas, (dot_x + i*30, h-badge_h+30), 10, dc, -1)
                cv2.putText(canvas, str(i+1), (dot_x + i*30 - 4, h-badge_h+35),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0,0,0), 1, cv2.LINE_AA)

            # pulsing alert ring for dangerous levels
            if current_level >= 2:
                pulse = int((time.time() * 4) % 2)
                if pulse == 0:
                    cv2.rectangle(canvas, (3, 3), (w-3, h-3), color, 2)

            # ════════════════════════════════════════
            #  DRAW RIGHT PANEL
            # ════════════════════════════════════════
            px = w + 12   # panel x start
            py = 0        # panel y cursor

            # panel background
            cv2.rectangle(canvas, (w, 0), (w+PANEL_W, h), (18,18,18), -1)
            cv2.line(canvas, (w, 0), (w, h), (60,60,60), 1)

            # ── LIVE METRICS ──
            py = 30
            cv2.putText(canvas, "LIVE METRICS", (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160,160,160), 1, cv2.LINE_AA)
            cv2.line(canvas, (px, py+6), (w+PANEL_W-12, py+6), (50,50,50), 1)
            py += 30

            draw_bar(canvas, px, py, 200, 14, avg_ear_val, 0.5,
                     (50,205,50) if avg_ear_val > 0.20 else (0,80,220), "EAR (eye openness)")
            py += 40
            draw_bar(canvas, px, py, 200, 14, mar_val, 1.0,
                     (220,140,0) if mar_val > 0.5 else (100,180,100), "MAR (mouth/yawn)")
            py += 40

            # pitch bar
            pitch_val = abs(feats[4]) if detected else 0.0
            draw_bar(canvas, px, py, 200, 14, min(pitch_val, 30), 30,
                     (0,140,255) if pitch_val > 15 else (100,160,100), "Head pitch (nod)")
            py += 40

            # ── EAR MINI GRAPH ──
            py += 10
            cv2.putText(canvas, "EAR HISTORY", (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160,160,160), 1, cv2.LINE_AA)
            py += 8
            graph_h = 60
            graph_w = PANEL_W - 24
            cv2.rectangle(canvas, (px, py), (px+graph_w, py+graph_h), (35,35,35), -1)
            cv2.rectangle(canvas, (px, py), (px+graph_w, py+graph_h), (60,60,60), 1)

            # threshold line at EAR=0.20
            thresh_y = py + graph_h - int(0.20 / 0.5 * graph_h)
            cv2.line(canvas, (px, thresh_y), (px+graph_w, thresh_y), (0,80,200), 1)
            cv2.putText(canvas, "0.20", (px+graph_w-32, thresh_y-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0,120,255), 1, cv2.LINE_AA)

            if len(ear_history) > 1:
                pts = list(ear_history)
                for i in range(1, len(pts)):
                    x1g = px + int((i-1) / 60 * graph_w)
                    x2g = px + int(i     / 60 * graph_w)
                    y1g = py + graph_h - int(min(pts[i-1], 0.5) / 0.5 * graph_h)
                    y2g = py + graph_h - int(min(pts[i],   0.5) / 0.5 * graph_h)
                    gc  = (50,205,50) if pts[i] > 0.20 else (0,80,220)
                    cv2.line(canvas, (x1g, y1g), (x2g, y2g), gc, 1, cv2.LINE_AA)
            py += graph_h + 20

            # ── SESSION STATS ──
            cv2.putText(canvas, "SESSION STATS", (px, py),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.50, (160,160,160), 1, cv2.LINE_AA)
            cv2.line(canvas, (px, py+6), (w+PANEL_W-12, py+6), (50,50,50), 1)
            py += 25

            total_t = sum(level_times) + 1e-6
            for i, (lname, lt) in enumerate(zip(
                ["Alert", "Microsleep", "Drowsy", "Extreme"],
                level_times
            )):
                lc  = LEVELS[i]["color"]
                pct = lt / total_t
                bw  = int(pct * (PANEL_W - 100))

                cv2.putText(canvas, lname, (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, lc, 1, cv2.LINE_AA)
                cv2.rectangle(canvas, (px+90, py-10), (px+90+max(bw,2), py-2), lc, -1)
                cv2.putText(canvas, format_time(lt), (px+220, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.34, (140,140,140), 1, cv2.LINE_AA)
                py += 22

            # ── CONTROLS ──
            py = h - 80
            cv2.line(canvas, (px, py), (w+PANEL_W-12, py), (50,50,50), 1)
            py += 18
            for line in ["Q  — quit", "R  — reset session stats"]:
                cv2.putText(canvas, line, (px, py),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, (100,100,100), 1, cv2.LINE_AA)
                py += 18

            # ── show ──
            cv2.imshow("Driver Drowsiness Detection", canvas)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), ord("Q")):
                break
            elif key in (ord("r"), ord("R")):
                level_times  = [0.0, 0.0, 0.0, 0.0]
                level_start  = time.time()
                session_start = time.time()
                pred_buffer.clear()
                ear_history.clear()
                print("  Session stats reset.")

            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*55)
    print("  SESSION SUMMARY")
    print("="*55)
    for i, (lname, lt) in enumerate(zip(
        ["Alert", "Microsleep", "Drowsy", "Extreme Fatigue"], level_times
    )):
        print(f"  {lname:<20} {format_time(lt)}")
    print(f"  Total session     {format_time(sum(level_times))}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()