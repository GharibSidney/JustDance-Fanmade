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
from values import BODY_SKELETON

# ============================================================
# 🔹 CONFIG
# ============================================================

VIDEO_PATH = "musics/Just Dance 2017 PC Unlimited Rasputin 4K.webm"   # change to your video

device = Accelerator().device

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
# 🔹 SUPERVISION ANNOTATORS
# ============================================================

# edge_annotator = sv.EdgeAnnotator(color=sv.Color.GREEN, thickness=1)
edge_annotator = sv.EdgeAnnotator(
    edges=BODY_SKELETON,
    color=sv.Color.GREEN,
    thickness=2
)
vertex_annotator = sv.VertexAnnotator(color=sv.Color.RED, radius=2)


# ============================================================
# 🔹 OPEN VIDEO
# ============================================================

cap = cv2.VideoCapture(VIDEO_PATH)

print("🚀 Starting live pose estimation...")
print("Press 'Q' to quit.\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]

    # Convert to PILq
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

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

            xy_body = xy[:, 5:, :]
            scores_body = scores[:, 5:]

            key_points = sv.KeyPoints(
                xy=xy_body,
                confidence=scores_body
            )

            frame = edge_annotator.annotate(scene=frame, key_points=key_points)
            frame = vertex_annotator.annotate(scene=frame, key_points=key_points)

    # --------------------------------------------------------
    # Show Live Window
    # --------------------------------------------------------
    cv2.imshow("Live Pose Estimation", frame)

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cap.release()
cv2.destroyAllWindows()

print("✅ Finished.")