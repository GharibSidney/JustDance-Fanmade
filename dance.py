import cv2
import numpy as np
from ultralytics import YOLO
from constantes import set_song
from utils import run_audio, get_labels, get_scale, draw_skeleton,run_video, run_webcam
from score import get_smoothed_score

model = YOLO("yolo26n-pose.pt")
# TODO to change for argument
set_song("Rasputin")
cap = run_webcam()
video_cap = run_video()
# Webcam

# frame_time = 1.0 / fps
labels = get_labels()
score_buffer = []
run_audio()
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
    if frame_index % 9 == 0:
        results = model(frame, device="cpu", verbose=False)

        if results[0].keypoints is not None:

            yolo_kpts = results[0].keypoints.xy.cpu().numpy()
            yolo_conf = results[0].keypoints.conf.cpu().numpy()

            for person_kpts, person_conf in zip(yolo_kpts[:1], yolo_conf[:1]): # for many people if I eventually code it, for now only one person

            # Require hips and shoulders detected
                if (
                    person_conf[11] < 0.3 or person_conf[12] < 0.3 or
                    person_conf[5] < 0.3  or person_conf[6] < 0.3
                ):
                    continue

                scale, hip_center_x, hip_center_y = get_scale(person_kpts, )
                # Draw YOLO hip center
                cv2.circle(frame, (int(hip_center_x), int(hip_center_y)), 8, (0,255,255), -1)

                data = labels.get(frame_index)
                if data is None or len(data["persons"]) == 0:
                    continue
                label_person = data["persons"][0]  # taking first person only for now
                keypoints = label_person["keypoints_to_hips_normalized"]
                confidences = label_person["confidence"][0]

                # Re-project label keypoints using YOLO hip center
                labeled_projected_points = []
                for (dx, dy), c in zip(keypoints, confidences):
                    # if c > 0.2:
                    x = hip_center_x + dx * scale #* img_w
                    y = hip_center_y + dy * scale #* img_h
                    labeled_projected_points.append((x, y, c))
                    cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)
                    # else:
                    #     labeled_projected_points.append(None)
                draw_skeleton(labeled_projected_points, frame)
                get_smoothed_score([(yolo_kpts[0], yolo_conf[0])], labeled_projected_points, score_buffer)
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