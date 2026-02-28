import os
import json
import cv2
from values import BODY_SKELETON, MAX_FRAMES
# =========================
# Paths
# =========================
video_path = "musics/Just Dance 2017 PC Unlimited Rasputin 4K.webm"          # original video
labels_dir = "labels"             # folder containing json files
output_path = "output_annotated.mp4"

# =========================
# Open video
# =========================
cap = cv2.VideoCapture(video_path)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

frame_index = 0

# =========================
# Skeleton (optional)
# =========================


# =========================
# Process Video
# =========================
frame_index = 0

while frame_index < MAX_FRAMES:
    ret, frame = cap.read()
    if not ret:
        break

    label_path = os.path.join(labels_dir, f"{frame_index}.json")

    if os.path.exists(label_path):
        with open(label_path, "r") as f:
            data = json.load(f)

        for person in data["persons"]:

            bbox = person["bounding_box"]

            x1 = int(bbox["x1"])
            y1 = int(bbox["y1"])
            x2 = int(bbox["x2"])
            y2 = int(bbox["y2"])

            if x2 < x1:
                x2 = x1 + x2
            if y2 < y1:
                y2 = y1 + y2

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)

            keypoints = person["keypoints"][0]
            confidences = person["confidence"][0]

            for (x, y), conf in zip(keypoints, confidences):
                if conf > 0.2:
                    cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)

            for i_k, j_k in BODY_SKELETON:
                if confidences[i_k] > 0.2 and confidences[j_k] > 0.2:
                    x1_k, y1_k = keypoints[i_k]
                    x2_k, y2_k = keypoints[j_k]
                    cv2.line(frame,
                             (int(x1_k), int(y1_k)),
                             (int(x2_k), int(y2_k)),
                             (255, 0, 0),
                             2)

    # SHOW FRAME
    cv2.imshow("Preview", frame)

    # 30ms delay (adjust if too fast)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()