import cv2
import numpy as np
from ultralytics import YOLO
from constantes import set_song
from utils import get_labels, run_video


# ---------------- OKS ----------------
def compute_oks(pred_kpts, gt_kpts, bbox, sigmas):
    bbox_w = abs(bbox["x2"] - bbox["x1"])
    bbox_h = abs(bbox["y2"] - bbox["y1"])
    s = bbox_w * bbox_h

    if s < 1e-6:
        return 0.0

    vars = (sigmas * 2) ** 2

    oks = 0.0
    for i in range(len(gt_kpts)):
        dx = pred_kpts[i][0] - gt_kpts[i][0]
        dy = pred_kpts[i][1] - gt_kpts[i][1]
        e = (dx**2 + dy**2) / (2 * s * vars[i])
        oks += np.exp(-e)

    return oks / len(gt_kpts)

def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter_area = inter_w * inter_h

    areaA = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    areaB = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))

    union = areaA + areaB - inter_area

    if union == 0:
        return 0.0

    return inter_area / union


# COCO-like sigmas adapted to your 12 joints
SIGMAS = np.array([
    0.26, 0.26,  # shoulders
    0.25, 0.25,  # elbows
    0.35, 0.35,  # wrists
    0.26, 0.26,  # hips
    0.25, 0.25,  # knees
    0.35, 0.35   # ankles
])


def main(song: str = "Rasputin"):

    set_song(song)

    model = YOLO("yolo26n-pose.pt")
    video_cap, fps = run_video()
    labels = get_labels()

    frame_index = 0

    # ---------------- Accumulators ----------------
    total_correct_pck = 0
    total_joints = 0
    oks_scores = []

    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        results = model(frame, device="cuda", verbose=False)
        data = labels.get(frame_index)

        if data is None or len(data["persons"]) == 0:
            frame_index += 1
            continue

        if results[0].keypoints is not None:

            yolo_kpts = results[0].keypoints.xy.cpu().numpy()
            yolo_boxes = results[0].boxes.xyxy.cpu().numpy()
            for person_kpts in yolo_kpts[:1]:  # single person
                
                gt_person = data["persons"][0]
                gt_kpts = np.array(gt_person["keypoints"][0])  # (12,2)
                gt_box_raw = gt_person["bounding_box"]

                # YOLO remove head (COCO index 0–4)
                pred_kpts = person_kpts[5:5+len(gt_kpts)]
                pred_box = yolo_boxes[0]  # first detected person
                # PCK (torso)
                left_shoulder = gt_kpts[0]
                right_shoulder = gt_kpts[1]
                left_hip = gt_kpts[6]
                right_hip = gt_kpts[7]

                shoulder_center = 0.5 * (left_shoulder + right_shoulder)
                hip_center = 0.5 * (left_hip + right_hip)

                torso_size = np.linalg.norm(shoulder_center - hip_center)

                if torso_size > 1e-6:
                    for p, g in zip(pred_kpts, gt_kpts):
                        dist = np.linalg.norm(p - g)
                        if dist < 0.5 * torso_size:
                            total_correct_pck += 1

                    total_joints += len(gt_kpts)
                gt_box = [
                min(gt_box_raw["x1"], gt_box_raw["x2"]),
                min(gt_box_raw["y1"], gt_box_raw["y2"]),
                max(gt_box_raw["x1"], gt_box_raw["x2"]),
                max(gt_box_raw["y1"], gt_box_raw["y2"]), ]

                iou = compute_iou(pred_box, gt_box)
                # ---------------- OKS ----------------
                oks = compute_oks(
                    pred_kpts,
                    gt_kpts,
                    gt_person["bounding_box"],
                    SIGMAS
                )
                
                oks_scores.append(oks)
                try:
                    iou_scores.append(iou)
                except NameError:
                    iou_scores = [iou]

        frame_index += 1

    video_cap.release()

    # Final Metrics 

    # PCK
    pck = total_correct_pck / total_joints if total_joints > 0 else 0

    # mAP (OKS)
    thresholds = np.arange(0.5, 1.0, 0.05)
    ap_list = []

    for t in thresholds:
        correct = sum(oks >= t for oks in oks_scores)
        ap = correct / len(oks_scores) if len(oks_scores) > 0 else 0
        ap_list.append(ap)

    mAP = np.mean(ap_list) if len(ap_list) > 0 else 0
    # mAP (IoU) 
    thresholds = np.arange(0.5, 1.0, 0.05)
    ap_list_bbox = []

    for t in thresholds:
        correct = sum(iou >= t for iou in iou_scores)
        ap = correct / len(iou_scores) if len(iou_scores) > 0 else 0
        ap_list_bbox.append(ap)

    mAP_bbox = np.mean(ap_list_bbox) if len(ap_list_bbox) > 0 else 0

    print(f"AP_list (bbox): {ap_list_bbox}")
    print(f"mAP (bbox IoU): {mAP_bbox:.4f}")
    print(f"PCK (torso): {pck:.4f}")
    print(f"AP_list (OKS): {ap_list}")
    print(f"mAP (OKS): {mAP:.4f}")


if __name__ == "__main__":
    main()