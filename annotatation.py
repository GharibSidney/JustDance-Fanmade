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
from constantes import VIDEO_PATH, MAX_FRAMES 


# ============================================================
# 🔹 CONFIG
# ============================================================

  # change to your video

device = Accelerator().device

labels_dir = "labels"
os.makedirs(labels_dir, exist_ok=True)
# ============================================================
# 🔹 LOAD MODELS (ONLY ONCE)
# ============================================================

person_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

pose_processor = AutoProcessor.from_pretrained("usyd-community/vitpose-base-simple")
pose_model = VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple", device_map=device)

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

    results = person_processor.post_process_object_detection(outputs, target_sizes=torch.tensor([(height, width)]))[0]

    person_boxes = results["boxes"][results["labels"] == 0]

    if len(person_boxes) > 0:
        person_boxes = person_boxes.cpu().numpy()

        # Convert VOC → COCO (x, y, w, h)
        person_boxes[:, 2] -= person_boxes[:, 0]
        person_boxes[:, 3] -= person_boxes[:, 1]

        # --------------------------------------------------------
        # 2️⃣ Pose Estimation
        # --------------------------------------------------------
        pose_inputs = pose_processor(image, boxes=[person_boxes], return_tensors="pt").to(device)

        with torch.no_grad():
            pose_outputs = pose_model(**pose_inputs)

        pose_results = pose_processor.post_process_pose_estimation(pose_outputs,boxes=[person_boxes])[0]

        if len(pose_results) > 0:
            xy = torch.stack([pose_result["keypoints"]for pose_result in pose_results]).cpu().numpy()

            scores = torch.stack([pose_result["scores"] for pose_result in pose_results]).cpu().numpy()

            key_points = sv.KeyPoints(xy=xy[:, 5:, :], confidence=scores[:, 5:])
            key_points_normalized = key_points.xy.copy()
            key_points_normalized[..., 0] /= width
            key_points_normalized[..., 1] /= height
            x1 = person_boxes[0, 0]
            y1 = person_boxes[0, 1]
            # 1️⃣ Center on hips
            kp = key_points.xy[0] 
            hip_center = (kp[6] + kp[7]) / 2
            kp_centered = kp - hip_center
            kp_centered_normalized = kp_centered.copy()
            kp_centered_normalized[..., 0] /= width
            kp_centered_normalized[..., 1] /= height
            hip_center_normalized = hip_center.copy()
            hip_center_normalized[0] /= width
            hip_center_normalized[1] /= height
            # box_width = person_boxes[0, 2] #x2
            # box_height = person_boxes[0, 3] #y2
            # box_width = max(box_width, 1e-6) # for security
            # box_height = max(box_height, 1e-6)

            # Copy keypoints
            # keypoints_bbox_normalized = key_points.xy.copy()

            # Normalize relative to bounding box
            # keypoints_bbox_normalized[..., 0] = (keypoints_bbox_normalized[..., 0] - x1) / box_width
            # keypoints_bbox_normalized[..., 1] = (keypoints_bbox_normalized[..., 1] - y1) / box_height
            person_data = {
            "bounding_box": {
                "x1": float(person_boxes[0, 0]),
                "y1": float(person_boxes[0, 1]),
                "x2": float(person_boxes[0, 2]),
                "y2": float(person_boxes[0, 3])
            },
            "bounding_box_normalized":{
            "x1": float(person_boxes[0, 0]) / width,
            "y1": float(person_boxes[0, 1]) / height,
            "x2": float(person_boxes[0, 2]) / width,
            "y2": float(person_boxes[0, 3]) / height
            },
            "keypoints":  key_points.xy.tolist(),
            "confidence": key_points.confidence.tolist(),
            # "keypoints_normalized":key_points_normalized.tolist(),
            # "hip_center": hip_center.tolist(),
            # "keypoints_to_hips": kp_centered.tolist(),
            "keypoints_to_hips_normalized": kp_centered_normalized.tolist(),
            "hip_center_normalized": hip_center_normalized.tolist()
        }

        label_data["persons"].append(person_data)
    try:
        if label_data["persons"] == [] and person_data:
            # If no detection were found, I just take the last frame as label
            # It makes the detection more fluid
            label_data["persons"].append(person_data)
    except:
        pass
    label_path = os.path.join(labels_dir, f"{frame_index}.json")
    with open(label_path, "w") as f:
        json.dump(label_data, f, indent=2)
    frame_index+=1


cap.release()

print("✅ Finished.")