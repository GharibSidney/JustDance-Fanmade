import cv2
import numpy as np
from ultralytics import YOLO

# =========================
# Load YOLO Pose Model
# =========================
model = YOLO("yolo26n-pose.pt")

# =========================
# Open LIVE webcam
# =========================
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

frame_index = 0

# COCO Skeleton connections (YOLO default)
SKELETON = [
    (5, 6), (5, 7), (7, 9),
    (6, 8), (8, 10),
    (5, 11), (6, 12),
    (11, 12),
    (11, 13), (13, 15),
    (12, 14), (14, 16)
]

# Head indices to ignore
HEAD_POINTS = {0, 1, 2, 3, 4}

# =========================
# Process LIVE Camera
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, device="cpu", verbose=False)
    annotated_frame = frame.copy()

    if results[0].keypoints is not None:

        keypoints = results[0].keypoints.xy.cpu().numpy()
        confidences = results[0].keypoints.conf.cpu().numpy()

        for person_kpts, person_conf in zip(keypoints, confidences):

            # =========================
            # Draw BODY keypoints only
            # =========================
            for i, (x, y) in enumerate(person_kpts):
                if i in HEAD_POINTS:
                    continue
                if person_conf[i] > 0.3:
                    cv2.circle(annotated_frame, (int(x), int(y)), 6, (0, 0, 255), -1)

            # =========================
            # Draw BODY skeleton only
            # =========================
            for i, j in SKELETON:
                if i in HEAD_POINTS or j in HEAD_POINTS:
                    continue
                if person_conf[i] > 0.3 and person_conf[j] > 0.3:
                    x1, y1 = person_kpts[i]
                    x2, y2 = person_kpts[j]
                    cv2.line(annotated_frame, (int(x1), int(y1)),(int(x2), int(y2)), (255, 0, 0), 2, )

    cv2.imshow("Live YOLO Pose (No Head)", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()