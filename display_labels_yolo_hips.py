import cv2
import numpy as np
from ultralytics import YOLO
from constantes import BODY_SKELETON, VIDEO_PATH, AUDIO_PATH
from utils import get_label_torso, get_audio, get_labels

model = YOLO("yolo26n-pose.pt")

label_torso = get_label_torso()
# Webcam
cap = cv2.VideoCapture(0) # live video
video_cap = cv2.VideoCapture(VIDEO_PATH) # dance video
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()
# fps = video_cap.get(cv2.CAP_PROP_FPS)
# frame_time = 1.0 / fps
labels = get_labels(labels_dir="labels")

get_audio(AUDIO_PATH)
frame_index = 0
# start_time = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame = cv2.flip(frame, 1)  # mirror effect
    # video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index) # Set the frame according to the fps to sync audio and video
    ret2, video_frame = video_cap.read()
    if not ret2:
        break

    # YOLO Pose
    results = model(frame, device="cpu", verbose=False)

    if results[0].keypoints is not None:

        yolo_kpts = results[0].keypoints.xy.cpu().numpy()
        yolo_conf = results[0].keypoints.conf.cpu().numpy()

        for person_kpts, person_conf in zip(yolo_kpts, yolo_conf):

           # Require hips and shoulders detected
            if (
                person_conf[11] < 0.3 or person_conf[12] < 0.3 or
                person_conf[5] < 0.3  or person_conf[6] < 0.3
            ):
                continue

            # Compute YOLO hip center
            hip_center_x = (person_kpts[11][0] + person_kpts[12][0]) / 2
            hip_center_y = (person_kpts[11][1] + person_kpts[12][1]) / 2

            # Compute YOLO shoulder center
            shoulder_center_x = (person_kpts[5][0] + person_kpts[6][0]) / 2
            shoulder_center_y = (person_kpts[5][1] + person_kpts[6][1]) / 2
            # Convert to vectors
            hip_center = np.array([hip_center_x, hip_center_y])
            shoulder_center = np.array([shoulder_center_x, shoulder_center_y])
            # Compute torso length (pixels)
            player_torso = np.linalg.norm(shoulder_center - hip_center)
            # Compute scale
            scale = player_torso / label_torso
            # Draw YOLO hip center
            cv2.circle(frame, (int(hip_center_x), int(hip_center_y)), 8, (0,255,255), -1)

            data = labels.get(frame_index)
            if data is None or len(data["persons"]) == 0:
                continue
            label_person = data["persons"][0]  # taking first person
            keypoints = label_person["keypoints_to_hips_normalized"]
            confidences = label_person["confidence"][0]

            # Re-project label keypoints using YOLO hip center
            projected_points = []
            for (dx, dy), c in zip(keypoints, confidences):
                if c > 0.2:
                    x = hip_center_x + dx * scale #* img_w
                    y = hip_center_y + dy * scale #* img_h
                    projected_points.append((x, y))
                    cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
                else:
                    projected_points.append(None)
            # Draw Skeleton
            for i_k, j_k in BODY_SKELETON:
                if (
                    projected_points[i_k] is not None
                    and projected_points[j_k] is not None
                ):
                    x1, y1 = projected_points[i_k]
                    x2, y2 = projected_points[j_k]
                    cv2.line( frame, (int(x1), int(y1)), (int(x2), int(y2)),
                            (255, 0, 0), 2, )
    # Resize video to match webcam height
    h = frame.shape[0]
    video_frame = cv2.resize(video_frame, (int(video_frame.shape[1] * h / video_frame.shape[0]), h))

    # Combine side-by-side
    debug_view = np.hstack((frame, video_frame))

    cv2.imshow("Debug View (Player | Original)", debug_view)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # elapsed = time.time() - start_time
    # frame_index = int(elapsed * fps)
    frame_index+=1

cap.release()
video_cap.release()
cv2.destroyAllWindows()