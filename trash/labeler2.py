import os
import cv2
import torch
import numpy as np
import json
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation
)
from accelerate import Accelerator


# ============================================================
# 🔹 CONFIG
# ============================================================

video_path = "musics/Just Dance 2017 PC Unlimited Rasputin 4K.webm"
labels_dir = "labels"
confidence_threshold = 0.3

os.makedirs(labels_dir, exist_ok=True)

device = Accelerator().device


# ============================================================
# 🔹 LOAD MODELS
# ============================================================

# ---- Person Detector (RT-DETR) ----
person_processor = AutoProcessor.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365"
)

person_model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365",
    device_map=device
)

# ---- Pose Model (ViTPose) ----
pose_processor = AutoProcessor.from_pretrained(
    "usyd-community/vitpose-base-simple"
)

pose_model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-base-simple",
    device_map=device
)


# ============================================================
# 🔹 COCO KEYPOINT NAMES
# ============================================================

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]


# ============================================================
# 🔹 PROCESS VIDEO
# ============================================================

cap = cv2.VideoCapture(video_path)
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_index > 90:
        break
    print(frame_index)
    # --------------------------------------------------------
    # 1️⃣ Convert frame to PIL
    # --------------------------------------------------------
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)
    h, w = image.height, image.width

    # --------------------------------------------------------
    # 2️⃣ Detect Persons
    # --------------------------------------------------------
    inputs = person_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(h, w)]),
        threshold=confidence_threshold
    )[0]

    # Keep only class 0 (person)
    person_mask = results["labels"] == 0
    person_boxes = results["boxes"][person_mask]

    if len(person_boxes) == 0:
        frame_index += 1
        continue

    person_boxes = person_boxes.cpu().numpy()

    # Keep largest box
    areas = (person_boxes[:, 2] - person_boxes[:, 0]) * \
            (person_boxes[:, 3] - person_boxes[:, 1])

    largest_idx = np.argmax(areas)
    person_boxes = person_boxes[largest_idx:largest_idx + 1]

    # --------------------------------------------------------
    # 3️⃣ Run Pose Estimation
    # --------------------------------------------------------
    pose_inputs = pose_processor(
        image,
        boxes=[person_boxes],
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        pose_outputs = pose_model(**pose_inputs)

    pose_results = pose_processor.post_process_pose_estimation(
        pose_outputs,
        boxes=[person_boxes]
    )[0]

    # --------------------------------------------------------
    # 4️⃣ Build JSON Structure
    # --------------------------------------------------------
    label_data = {
        "frame_index": frame_index,
        "image_width": w,
        "image_height": h,
        "persons": []
    }

    for pose_result in pose_results:

        keypoints = pose_result["keypoints"].cpu().numpy()
        scores = pose_result["scores"].cpu().numpy()

        # --------------------------
        # Normalize to [0,1]
        # --------------------------
        keypoints_norm = keypoints.copy()
        keypoints_norm[:, 0] /= w
        keypoints_norm[:, 1] /= h

        # --------------------------
        # Center on hip midpoint
        # --------------------------
        left_hip = keypoints_norm[11]
        right_hip = keypoints_norm[12]
        hip_center = (left_hip + right_hip) / 2.0

        if not (0 < hip_center[0] < 1 and 0 < hip_center[1] < 1):
            bbox = person_boxes[0]
            hip_center = np.array([
                ((bbox[0] + bbox[2]) / 2) / w,
                ((bbox[1] + bbox[3]) / 2) / h
            ])

        keypoints_centered = keypoints_norm - hip_center

        # --------------------------
        # Scale normalization
        # --------------------------
        shoulder_mid = (
            keypoints_centered[5] + keypoints_centered[6]
        ) / 2.0

        torso_length = np.linalg.norm(shoulder_mid)

        if torso_length < 0.01:
            bbox = person_boxes[0]
            torso_length = (bbox[3] - bbox[1]) / h

        keypoints_scaled = (
            keypoints_centered / torso_length
            if torso_length > 1e-6
            else keypoints_centered
        )

        # --------------------------
        # Build keypoint JSON
        # --------------------------
        keypoints_raw_json = []
        keypoints_norm_json = []

        for i in range(len(KEYPOINT_NAMES)):
            if scores[i] < confidence_threshold:
                x_raw, y_raw = 0.0, 0.0
                x_norm, y_norm = 0.0, 0.0
            else:
                x_raw, y_raw = keypoints[i]
                x_norm, y_norm = keypoints_scaled[i]

            keypoints_raw_json.append({
                "name": KEYPOINT_NAMES[i],
                "x": float(x_raw),
                "y": float(y_raw),
                "score": float(scores[i])
            })

            keypoints_norm_json.append({
                "name": KEYPOINT_NAMES[i],
                "x": float(x_norm),
                "y": float(y_norm),
                "score": float(scores[i])
            })

        # --------------------------
        # Person JSON
        # --------------------------
        person_data = {
            "bounding_box": {
                "x1": float(person_boxes[0, 0]),
                "y1": float(person_boxes[0, 1]),
                "x2": float(person_boxes[0, 2]),
                "y2": float(person_boxes[0, 3])
            },
            "keypoints": keypoints_raw_json,
            "keypoints_normalized": keypoints_norm_json,
            "normalization": {
                "hip_center_x": float(hip_center[0]),
                "hip_center_y": float(hip_center[1]),
                "torso_length": float(torso_length)
            }
        }

        label_data["persons"].append(person_data)

    # --------------------------------------------------------
    # 5️⃣ Save JSON
    # --------------------------------------------------------
    label_path = os.path.join(labels_dir, f"{frame_index}.json")

    with open(label_path, "w") as f:
        json.dump(label_data, f, indent=2)

    frame_index += 1


cap.release()
print(f"✅ Done. Saved {frame_index} label files in '{labels_dir}'")