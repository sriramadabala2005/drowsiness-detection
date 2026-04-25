import cv2
import mediapipe as mp
import numpy as np
import os
import csv
import time

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_CSV  = os.path.join(BASE_DIR, "features.csv")
MODEL_PATH  = os.path.join(BASE_DIR, "scripts", "face_landmarker.task")

CLASSES = ["alert", "microsleep", "drowsy", "extreme_fatigue"]
LABEL_MAP = {"alert": 0, "microsleep": 1, "drowsy": 2, "extreme_fatigue": 3}

# ─────────────────────────────────────────────
#  MEDIAPIPE SETUP
# ─────────────────────────────────────────────
BaseOptions           = mp.tasks.BaseOptions
FaceLandmarker        = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.4,
    min_face_presence_confidence=0.4,
    min_tracking_confidence=0.4,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False,
)

# ─────────────────────────────────────────────
#  LANDMARK INDICES
# ─────────────────────────────────────────────

# Eye landmarks (dlib-style mapped to MediaPipe 468 mesh)
# Left eye
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye
RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# Mouth landmarks
MOUTH = [61, 291, 39, 181, 0, 17, 269, 405]

# Head pose reference points (nose tip, chin, left eye corner, right eye corner, left mouth, right mouth)
HEAD_POSE_POINTS = [1, 152, 226, 446, 57, 287]

# ─────────────────────────────────────────────
#  FEATURE FUNCTIONS
# ─────────────────────────────────────────────

def get_landmark_coords(landmarks, indices, img_w, img_h):
    """Extract (x, y) pixel coords for given landmark indices."""
    coords = []
    for i in indices:
        lm = landmarks[i]
        coords.append((lm.x * img_w, lm.y * img_h))
    return np.array(coords, dtype=np.float64)


def eye_aspect_ratio(eye_coords):
    """
    EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
    Higher EAR = eyes more open
    Lower EAR  = eyes more closed (drowsy)
    Typical awake: ~0.25-0.35, drowsy: below 0.20
    """
    p1, p2, p3, p4, p5, p6 = eye_coords
    vertical1   = np.linalg.norm(p2 - p6)
    vertical2   = np.linalg.norm(p3 - p5)
    horizontal  = np.linalg.norm(p1 - p4)
    ear = (vertical1 + vertical2) / (2.0 * horizontal + 1e-6)
    return round(ear, 6)


def mouth_aspect_ratio(mouth_coords):
    """
    MAR = vertical mouth opening / horizontal mouth width
    Higher MAR = mouth more open (yawning)
    Typical closed: ~0.3, yawning: above 0.6
    """
    p1, p2, p3, p4, p5, p6, p7, p8 = mouth_coords
    vertical1  = np.linalg.norm(p3 - p7)
    vertical2  = np.linalg.norm(p4 - p8)
    vertical3  = np.linalg.norm(p5 - p6)
    horizontal = np.linalg.norm(p1 - p2)
    mar = (vertical1 + vertical2 + vertical3) / (3.0 * horizontal + 1e-6)
    return round(mar, 6)


def head_pose_angles(landmarks, img_w, img_h):
    """
    Estimate head pitch (up/down), yaw (left/right), roll (tilt)
    using solvePnP with 6 facial reference points.
    Returns (pitch, yaw, roll) in degrees.
    Pitch > 0  = head tilting down (drowsy nodding)
    """
    model_points = np.array([
        [0.0,    0.0,    0.0],      # nose tip
        [0.0,   -63.6,  -12.5],     # chin
        [-43.3,  32.7,  -26.0],     # left eye corner
        [43.3,   32.7,  -26.0],     # right eye corner
        [-28.9, -28.9,  -24.1],     # left mouth corner
        [28.9,  -28.9,  -24.1],     # right mouth corner
    ], dtype=np.float64)

    image_points = get_landmark_coords(landmarks, HEAD_POSE_POINTS, img_w, img_h)

    focal_length = img_w
    center       = (img_w / 2, img_h / 2)
    cam_matrix   = np.array([
        [focal_length, 0,            center[0]],
        [0,            focal_length, center[1]],
        [0,            0,            1         ],
    ], dtype=np.float64)
    dist_coeffs = np.zeros((4, 1))

    success, rot_vec, trans_vec = cv2.solvePnP(
        model_points, image_points, cam_matrix, dist_coeffs,
        flags=cv2.SOLVEPNP_ITERATIVE
    )

    if not success:
        return 0.0, 0.0, 0.0

    rot_mat, _ = cv2.Rodrigues(rot_vec)
    proj_mat   = np.hstack((rot_mat, trans_vec))
    _, _, _, _, _, _, euler = cv2.decomposeProjectionMatrix(proj_mat)

    pitch = float(euler[0])
    yaw   = float(euler[1])
    roll  = float(euler[2])
    return round(pitch, 4), round(yaw, 4), round(roll, 4)


def perclos_approx(ear, threshold=0.20):
    """
    Approximate PERCLOS from a single frame.
    Returns 1 if eye is considered closed (EAR below threshold), else 0.
    During training, we average this over a window later.
    """
    return 1 if ear < threshold else 0


def extract_features_from_image(img_path, landmarker):
    """
    Given an image path, returns a dict of features or None if face not found.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    img_h, img_w = img_bgr.shape[:2]
    img_rgb      = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mp_image     = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    result = landmarker.detect(mp_image)
    if not result.face_landmarks or len(result.face_landmarks) == 0:
        return None

    landmarks = result.face_landmarks[0]

    left_eye_coords  = get_landmark_coords(landmarks, LEFT_EYE,  img_w, img_h)
    right_eye_coords = get_landmark_coords(landmarks, RIGHT_EYE, img_w, img_h)
    mouth_coords     = get_landmark_coords(landmarks, MOUTH,      img_w, img_h)

    left_ear  = eye_aspect_ratio(left_eye_coords)
    right_ear = eye_aspect_ratio(right_eye_coords)
    avg_ear   = round((left_ear + right_ear) / 2.0, 6)
    mar       = mouth_aspect_ratio(mouth_coords)
    pitch, yaw, roll = head_pose_angles(landmarks, img_w, img_h)
    eye_closed = perclos_approx(avg_ear)

    return {
        "left_ear":   left_ear,
        "right_ear":  right_ear,
        "avg_ear":    avg_ear,
        "mar":        mar,
        "pitch":      pitch,
        "yaw":        yaw,
        "roll":       roll,
        "eye_closed": eye_closed,
    }


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
def main():
    print("\n" + "="*55)
    print("  FEATURE EXTRACTION  —  Drowsiness Detection Dataset")
    print("="*55)

    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: face_landmarker.task not found at:\n  {MODEL_PATH}")
        print("Run collect_data.py first — it downloads the model.")
        return

    fieldnames = [
        "filename", "class_name", "label",
        "left_ear", "right_ear", "avg_ear",
        "mar", "pitch", "yaw", "roll", "eye_closed"
    ]

    total_processed = 0
    total_skipped   = 0
    class_counts    = {}

    start_time = time.time()

    with FaceLandmarker.create_from_options(options) as landmarker:
        with open(OUTPUT_CSV, "w", newline="") as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()

            for class_name in CLASSES:
                class_dir = os.path.join(DATASET_DIR, class_name)
                label     = LABEL_MAP[class_name]

                if not os.path.exists(class_dir):
                    print(f"  WARNING: folder not found — {class_dir}")
                    continue

                images = [f for f in os.listdir(class_dir)
                          if f.lower().endswith((".jpg", ".jpeg", ".png"))]
                images.sort()

                processed = 0
                skipped   = 0

                print(f"\n  Processing [{class_name}] — {len(images)} images")

                for i, fname in enumerate(images):
                    img_path = os.path.join(class_dir, fname)
                    features = extract_features_from_image(img_path, landmarker)

                    if features is None:
                        skipped += 1
                        continue

                    row = {
                        "filename":   fname,
                        "class_name": class_name,
                        "label":      label,
                        **features
                    }
                    writer.writerow(row)
                    processed += 1

                    if (i + 1) % 50 == 0:
                        pct = int(((i+1) / len(images)) * 20)
                        bar = "█" * pct + "░" * (20 - pct)
                        print(f"    [{bar}] {i+1}/{len(images)}", end="\r")

                class_counts[class_name] = {"processed": processed, "skipped": skipped}
                total_processed += processed
                total_skipped   += skipped
                print(f"    Done — {processed} extracted, {skipped} skipped (no face detected)    ")

    elapsed = round(time.time() - start_time, 1)

    print("\n" + "="*55)
    print("  EXTRACTION COMPLETE")
    print("="*55)
    for cls, counts in class_counts.items():
        print(f"  {cls:<20}  extracted: {counts['processed']}   skipped: {counts['skipped']}")
    print(f"\n  Total rows in CSV : {total_processed}")
    print(f"  Total skipped     : {total_skipped}")
    print(f"  Time taken        : {elapsed}s")
    print(f"\n  Saved to: {OUTPUT_CSV}")
    print("="*55 + "\n")

    if total_processed < 100:
        print("  WARNING: Very few features extracted.")
        print("  Check that your dataset images contain visible faces.\n")
    else:
        print("  Ready for model training! Run train_model.py next.\n")


if __name__ == "__main__":
    main()
    