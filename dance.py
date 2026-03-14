import cv2
import numpy as np
from ultralytics import YOLO
from constantes import set_song, FRAME_TO_PREDICT
from utils import (
    run_audio, get_labels, get_scale, draw_skeleton,
    run_video, run_webcam, stop_audio, add_small_image_corner
)
from score import get_smoothed_score


def main(song: str = "Rasputin"):

    set_song(song)

    model = YOLO("yolo26n-pose.pt")

    video_cap, fps = run_video()
    cap = run_webcam()

    labels = get_labels()

    score_buffer = []
    frame_index = 0
    total_score = 0

    # score display control
    last_score_update = 0
    current_score_image = None
    fade_start_frame = None
    fade_duration = int(fps * 2)  # fade lasts 2 seconds

    run_audio()

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        ret2, video_frame = video_cap.read()
        if not ret2:
            break

        # YOLO pose detection
        if frame_index % FRAME_TO_PREDICT == 0:

            results = model(frame, device="cpu", verbose=False)

            if results[0].keypoints is not None:

                yolo_kpts = results[0].keypoints.xy.cpu().numpy()
                yolo_conf = results[0].keypoints.conf.cpu().numpy()

                for person_kpts, person_conf in zip(yolo_kpts[:1], yolo_conf[:1]):

                    # require hips + shoulders
                    if (
                        person_conf[11] < 0.3 or person_conf[12] < 0.3 or
                        person_conf[5] < 0.3 or person_conf[6] < 0.3
                    ):
                        continue

                    scale, hip_center_x, hip_center_y = get_scale(person_kpts)

                    cv2.circle(frame, (int(hip_center_x), int(hip_center_y)), 8, (0,255,255), -1)

                    data = labels.get(frame_index)

                    if data is None or len(data["persons"]) == 0:
                        continue

                    label_person = data["persons"][0]

                    keypoints = label_person["keypoints_to_hips_normalized"]
                    confidences = label_person["confidence"][0]

                    labeled_projected_points = []

                    for (dx, dy), c in zip(keypoints, confidences):

                        x = hip_center_x + dx * scale
                        y = hip_center_y + dy * scale

                        labeled_projected_points.append((x, y, c))

                        cv2.circle(frame, (int(x), int(y)), 6, (0, 0, 255), -1)

                    draw_skeleton(labeled_projected_points, frame)


                    # Score calculation
                    score_buffer, smoothed_score, image_score_path = get_smoothed_score([(yolo_kpts[0], yolo_conf[0])], labeled_projected_points,score_buffer)
                    total_score += smoothed_score

                    # Update score display every 4 seconds after 5 seconds
                    if frame_index > 5 * fps and frame_index - last_score_update >= fps * 4:

                        current_score_image = image_score_path
                        fade_start_frame = frame_index
                        last_score_update = frame_index

        # Score image fade animation

        if current_score_image is not None and fade_start_frame is not None:
            elapsed = frame_index - fade_start_frame
            alpha = max(0, 1 - elapsed / fade_duration)
            video_frame = add_small_image_corner(current_score_image, video_frame, alpha)

            if alpha == 0:
                current_score_image = None
                fade_start_frame = None

        # ------------------------------
        # Resize + display
        # ------------------------------

        h = frame.shape[0]

        video_frame = cv2.resize(video_frame,(int(video_frame.shape[1] * h / video_frame.shape[0]), h))
        debug_view = np.hstack((frame, video_frame))
        cv2.imshow("Debug View (Player | Original)", debug_view)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        frame_index += 1

    cap.release()
    video_cap.release()

    cv2.destroyAllWindows()

    stop_audio()

    print("FINAL SCORE:", int(total_score))


if __name__ == "__main__":
    main()