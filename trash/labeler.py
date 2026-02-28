import os
import cv2
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, RTDetrForObjectDetection, VitPoseForPoseEstimation
from accelerate import Accelerator

# =========================
# 🔹 PLACEHOLDER
# =========================
video_path = "downloads/Just Dance 2017 PC Unlimited Rasputin 4K.webm"

# =========================
# 🔹 Setup
# =========================
device = Accelerator().device
labels_dir = "labels"
os.makedirs(labels_dir, exist_ok=True)

# -------------------------
# Person detector (RT-DETR)
# -------------------------
person_image_processor = AutoProcessor.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365"
)
person_model = RTDetrForObjectDetection.from_pretrained(
    "PekingU/rtdetr_r50vd_coco_o365",
    device_map=device
)

# -------------------------
# VitPose
# -------------------------
pose_processor = AutoProcessor.from_pretrained(
    "usyd-community/vitpose-base-simple"
)
pose_model = VitPoseForPoseEstimation.from_pretrained(
    "usyd-community/vitpose-base-simple",
    device_map=device
)

# =========================
# 🔹 Open video
# =========================
cap = cv2.VideoCapture(video_path)
frame_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert BGR → RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(frame_rgb)

    # =========================
    # 1️⃣ Detect persons
    # =========================
    inputs = person_image_processor(images=image, return_tensors="pt").to(person_model.device)

    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs,
        target_sizes=torch.tensor([(image.height, image.width)]),
        threshold=0.3
    )[0]

    # Keep only person class (COCO class 0)
    person_boxes = results["boxes"][results["labels"] == 0]

    if len(person_boxes) == 0:
        frame_index += 1
        continue

    person_boxes = person_boxes.cpu().numpy()

    # Keep only largest bounding box
    areas = (person_boxes[:, 2] - person_boxes[:, 0]) * \
            (person_boxes[:, 3] - person_boxes[:, 1])

    largest_idx = np.argmax(areas)
    person_boxes = person_boxes[largest_idx:largest_idx+1]

    # =========================
    # 2️⃣ Run VitPose
    # =========================
    pose_inputs = pose_processor(
        image,
        boxes=[person_boxes],  # must be list of arrays
        return_tensors="pt"
    ).to(pose_model.device)

    with torch.no_grad():
        pose_outputs = pose_model(**pose_inputs)

    pose_results = pose_processor.post_process_pose_estimation(
        pose_outputs,
        boxes=[person_boxes]
    )[0]

    # =========================
    # 3️⃣ Save labels
    # =========================
    # Image size
    h, w = image.height, image.width

    label_path = os.path.join(labels_dir, f"{frame_index}.json")

    with open(label_path, "w") as f:
        for pose_result in pose_results:

            keypoints = pose_result["keypoints"]
            scores = pose_result["scores"]

            # Convert to numpy
            confidence_threshold = 0.3

            for i in range(len(keypoints)):
                if scores[i] < confidence_threshold:
                    keypoints[i] = torch.tensor([0.0, 0.0], device=keypoints.device)

            # -------------------------
            # 1️⃣ Normalize to [0,1]
            # -------------------------
            keypoints[:, 0] /= w
            keypoints[:, 1] /= h

            # -------------------------
            # 2️⃣ Center on hip midpoint
            # -------------------------
            left_hip = keypoints[11]
            right_hip = keypoints[12]
            hip_center = (left_hip + right_hip) / 2.0
            keypoints -= hip_center

            # -------------------------
            # 3️⃣ Scale normalization
            # -------------------------
            left_shoulder = keypoints[5]
            right_shoulder = keypoints[6]
            shoulder_mid = (left_shoulder + right_shoulder) / 2.0

            hip_mid = np.array([0.0, 0.0])  # after centering, hip center is at (0,0)

            torso_length = np.linalg.norm(shoulder_mid - hip_mid)

            if torso_length > 1e-6:
                keypoints /= torso_length

            # -------------------------
            # 4️⃣ Save only x,y
            # -------------------------
            for (x, y) in keypoints:
                f.write(f"{x:.6f} {y:.6f} ")

            f.write("\n")

    frame_index += 1

cap.release()

print(f"✅ Done. Saved {frame_index} label files in '{labels_dir}'")