import os
import json
import numpy as np
import cv2
import warnings
# disable annoying warnings
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
import pygame
from constantes import INDEX_LABEL_HIPS, BODY_SKELETON, VIDEO_PATH, AUDIO_PATH

def get_label_torso(labels_dir="labels"):

    label_path = os.path.join(labels_dir, f"{INDEX_LABEL_HIPS}.json")

    with open(label_path, "r") as f:
        data = json.load(f)

    label_person = data["persons"][0]  # assuming 1 person
    keypoints = label_person["keypoints_to_hips_normalized"]

    # shoulders (indices shifted because head removed)
    sx = (keypoints[0][0] + keypoints[1][0]) / 2
    sy = (keypoints[0][1] + keypoints[1][1]) / 2

    label_torso = np.sqrt(sx**2 + sy**2)
    # It is the distance between the hip center and
    # the shoulder center in the label coordinate system.
    return label_torso

def run_audio(audio_path:str=AUDIO_PATH):

    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

def run_video(video_path:str=VIDEO_PATH):
    video_cap = cv2.VideoCapture(video_path) # dance video
    if not video_cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    print("frame per second: ", fps)
    return video_cap

def run_webcam():
    cap = cv2.VideoCapture(0) # live video
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        exit()
    return cap

def get_labels(labels_dir="labels/Rasputin"):
    labels = {}
    for f in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, f)) as file:
            labels[int(f.split(".")[0])] = json.load(file)
    return labels

def get_scale(person_kpts):
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
    scale = player_torso / get_label_torso()
    return scale, hip_center_x, hip_center_y

def draw_skeleton(labeled_projected_points, frame):
    # Draw Skeleton
    for i_k, j_k in BODY_SKELETON:
        if (
            labeled_projected_points[i_k] is not None
            and labeled_projected_points[j_k] is not None
        ):
            x1, y1, _ = labeled_projected_points[i_k]
            x2, y2, _ = labeled_projected_points[j_k]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)),
                    (255, 0, 0), 2, )

if __name__ == "__main__":
    print(get_label_torso())