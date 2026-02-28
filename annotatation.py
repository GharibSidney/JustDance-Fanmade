import os
import json
import cv2
import torch
import numpy as np
import supervision as sv
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation
)
from accelerate import Accelerator
from values import MAX_FRAMES


# ============================================================
# 🔹 CONFIG
# ============================================================

VIDEO_PATH = "musics/Just Dance 2017 PC Unlimited Rasputin 4K.webm"   # change to your video

device = Accelerator().device

labels_dir = "labels"
os.makedirs(labels_dir, exist_ok=True)
# ============================================================
# 🔹 LOAD MODELS (ONLY ONCE)
# ============================================================

person_processor = AutoProcessor.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365"
)
person_model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365",
    device_map=device
)

pose_processor = AutoProcessor.from_pretrained(
    "usyd-community/vitpose-base-simple"
)
pose_model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-base-simple",
    device_map=device
)

# ============================================================
# 🔹 OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret or frame_index > MAX_FRAMES:
        break

    height, width = frame.shape[:2]

    # Convert to PILq
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

    label_data = {"frame_index": frame_index, "image_width": width, "image_height": height, "persons": []}
    # --------------------------------------------------------
    # 1️⃣ Person Detection
    # --------------------------------------------------------
    inputs = person_processor(images=image, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(height, width)])
    )[0]

    person_boxes = results["boxes"][results["labels"] == 0]

    if len(person_boxes) > 0:
        person_boxes = person_boxes.cpu().numpy()

        # Convert VOC → COCO (x, y, w, h)
        person_boxes[:, 2] -= person_boxes[:, 0]
        person_boxes[:, 3] -= person_boxes[:, 1]

        # --------------------------------------------------------
        # 2️⃣ Pose Estimation
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

        if len(pose_results) > 0:
            xy = torch.stack([
                pose_result["keypoints"]
                for pose_result in pose_results
            ]).cpu().numpy()

            scores = torch.stack([
                pose_result["scores"]
                for pose_result in pose_results
            ]).cpu().numpy()

            key_points = sv.KeyPoints(xy=xy[:, 5:, :], confidence=scores[:, 5:])
            key_points_normalized = key_points.xy.copy()
            key_points_normalized[..., 0] /= width
            key_points_normalized[..., 1] /= height
            person_data = {
            "bounding_box": {
                "x1": float(person_boxes[0, 0]),
                "y1": float(person_boxes[0, 1]),
                "x2": float(person_boxes[0, 2]),
                "y2": float(person_boxes[0, 3])
            },
            "keypoints":  key_points.xy.tolist(),
            "confidence": key_points.confidence.tolist(),
            "keypoints_normalized":key_points_normalized.tolist()
        }

        label_data["persons"].append(person_data)
    try:
        if label_data["persons"] == [] and person_data:
            # If no detection were found, I just take the last frame as label
            label_data["persons"].append(person_data)
    except:
        pass
    label_path = os.path.join(labels_dir, f"{frame_index}.json")
    with open(label_path, "w") as f:
        json.dump(label_data, f, indent=2)
    frame_index+=1


cap.release()

print("✅ Finished.")