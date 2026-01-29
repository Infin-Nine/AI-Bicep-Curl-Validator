import cv2
import math
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ------------------ HELPERS ------------------

def calculate_angle(a, b, c):
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]

    radians = math.atan2(c[1]-b[1], c[0]-b[0]) - \
              math.atan2(a[1]-b[1], a[0]-b[0])
    angle = abs(math.degrees(radians))
    return 360 - angle if angle > 180 else angle


def distance(a, b):
    return math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2)


def ema(new, old, alpha=0.25):
    if old is None:
        return new
    return alpha * new + (1 - alpha) * old

ORANGE = (0, 165, 255)
SKY_BLUE = (235, 206, 135)

model_path = "pose_landmarker_full.task" # Assumes the model is in the same folder

)

# ------------------ MEDIAPIPE ------------------

if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file not found: {model_path}. Please download it.")

base_options = python.BaseOptions(model_asset_path=model_path)
                                  

options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    num_poses=1,
    running_mode=vision.RunningMode.VIDEO
)

pose_detector = vision.PoseLandmarker.create_from_options(options)

# ------------------ VIDEO ------------------

cap = cv2.VideoCapture(0)
timestamp = 0

# ------------------ STATE ------------------

reps = 0
state = "down"
form_ok = True

# EMA values
smooth_angle = None
smooth_elbow_y = None
smooth_upper_arm = None

# Reference values
ref_elbow_y = None
ref_upper_arm = None

# ------------------ MAIN LOOP ------------------

POSE_CONNECTIONS = [
    # Upper body
    (0,5), (0,2),
    (8,5), (2,7),
    (10,9),

    (11, 13), (13, 15),      # Left arm
    (12, 14), (14, 16),      # Right arm
    (11, 12),                # Shoulders

    # Torso
    (11, 23), (12, 24),      # Shoulder → Hip
    (23, 24),                # Hip line

    # Lower body
    (23, 25), (25, 27),      # Left leg
    (24, 26), (26, 28),      # Right leg
]

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    timestamp += 33
    result = pose_detector.detect_for_video(mp_image, timestamp)

    if result.pose_landmarks:
        pose = result.pose_landmarks[0]

        # Right arm landmarks
        shoulder = pose[12]
        elbow = pose[14]
        wrist = pose[16]

        # Raw values
        raw_angle = calculate_angle(shoulder, elbow, wrist)
        raw_elbow_y = elbow.y
        raw_upper_arm = distance(shoulder, elbow)

        # EMA smoothing
        smooth_angle = ema(raw_angle, smooth_angle)
        smooth_elbow_y = ema(raw_elbow_y, smooth_elbow_y)
        smooth_upper_arm = ema(raw_upper_arm, smooth_upper_arm)

        # ---------------- FORM LOGIC ----------------

        # Capture reference in DOWN position
        if smooth_angle > 160:
            state = "down"
            ref_elbow_y = smooth_elbow_y
            ref_upper_arm = smooth_upper_arm
            form_ok = True

        # Validate form during curl
        if ref_elbow_y is not None:
            elbow_drift = abs(smooth_elbow_y - ref_elbow_y)
            upper_arm_change = abs(smooth_upper_arm - ref_upper_arm)

            if elbow_drift > 0.045 or upper_arm_change > 0.035:
                form_ok = False

        # Count valid rep
        if smooth_angle < 60 and state == "down" and form_ok:
            reps += 1
            state = "up"

        # ---------------- DRAW ----------------

        for idx, lm in enumerate(pose):
            x = int(lm.x * w)
            y = int(lm.y * h)

            color = ORANGE if idx % 2 == 0 else SKY_BLUE

            cv2.circle(
            frame,
            (x, y),
            6,
            color,
            -1
            )


        cv2.putText(frame, f"REPS: {reps}",
                    (30, 80), cv2.FONT_HERSHEY_SIMPLEX,
                    2, (0, 255, 0), 4)

        cv2.putText(frame, f"ANGLE: {int(smooth_angle)}",
                    (30, 140), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 255, 0), 2)

        cv2.putText(frame, f"FORM: {'OK' if form_ok else 'BAD'}",
                    (30, 190), cv2.FONT_HERSHEY_SIMPLEX,
                    1.2, (0,255,0) if form_ok else (0,0,255), 3)
        
        for start_idx, end_idx in POSE_CONNECTIONS:
            p1 = pose[start_idx]
            p2 = pose[end_idx]

            x1, y1 = int(p1.x * w), int(p1.y * h)
            x2, y2 = int(p2.x * w), int(p2.y * h)

            cv2.line(
                frame,
                (x1, y1),
                (x2, y2),
                (255, 255, 255),
                2
            )
        
        

    cv2.imshow("Biceps Curl (Form Validated)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
