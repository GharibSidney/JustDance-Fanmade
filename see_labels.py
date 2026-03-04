import os
import json
import cv2
from constantes import VIDEO_PATH, BODY_SKELETON, MAX_FRAMES
# =========================
# Paths
# =========================
labels_dir = "labels"             # folder containing json files

# =========================
# Open video
# =========================
cap = cv2.VideoCapture(VIDEO_PATH)

fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
frame_index = 0

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

            keypoints = person["keypoints_to_hips_normalized"]
            confidences = person["confidence"][0]
            hip_center = person["hip_center_normalized"]
            img_w = data["image_width"]
            img_h = data["image_height"]

            for (dx, dy), c in zip(keypoints, confidences):
                    if c > 0.2:
                        x = (hip_center[0] + dx) * img_w
                        y = (hip_center[1] + dy) * img_h
                        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
                
                # Draw skeleton lines
            for i_k, j_k in BODY_SKELETON:
                    if confidences[i_k] > 0.2 and confidences[j_k] > 0.2:
                        x1 = (hip_center[0] + keypoints[i_k][0]) * img_w
                        y1 = (hip_center[1] + keypoints[i_k][1]) * img_h
                        x2 = (hip_center[0] + keypoints[j_k][0]) * img_w
                        y2 = (hip_center[1] + keypoints[j_k][1]) * img_h
                        
                        cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)

    # SHOW FRAME
    cv2.imshow("Preview", frame)

    # 30ms delay (adjust if too fast)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

    frame_index += 1

cap.release()
cv2.destroyAllWindows()