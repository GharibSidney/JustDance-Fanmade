import os
import cv2
import json
from PIL import Image


# ============================================================
# 🔹 CONFIG
# ============================================================

VIDEO_PATH = "musics/Just Dance 2017 PC Unlimited Rasputin 4K.webm"
LABELS_DIR = "labels"
FRAME_INDEX = 89

CONF_THRESHOLD_HIGH = 0.5


# ============================================================
# 🔹 COCO FORMAT
# ============================================================

KEYPOINT_NAMES = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle"
]

SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16)
]


# ============================================================
# 🔹 LOAD LABEL
# ============================================================

def load_label(frame_index):
    path = os.path.join(LABELS_DIR, f"{frame_index}.json")

    if not os.path.exists(path):
        raise FileNotFoundError(f"Label file not found: {path}")

    with open(path, "r") as f:
        return json.load(f)


# ============================================================
# 🔹 LOAD VIDEO FRAME
# ============================================================

def load_frame(video_path, frame_index):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError(f"Cannot read frame {frame_index}")

    return frame


# ============================================================
# 🔹 DRAW FUNCTIONS
# ============================================================

def draw_bbox(image, bbox):
    x1, y1, x2, y2 = map(int, (bbox["x1"], bbox["y1"], bbox["x2"], bbox["y2"]))
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_skeleton(image, keypoints):
    for i, j in SKELETON:
        kp_i, kp_j = keypoints[i], keypoints[j]

        if is_zero(kp_i) or is_zero(kp_j):
            continue

        pt1 = (int(kp_i["x"]), int(kp_i["y"]))
        pt2 = (int(kp_j["x"]), int(kp_j["y"]))

        cv2.line(image, pt1, pt2, (255, 0, 0), 2)


def draw_keypoints(image, keypoints):
    for idx, kp in enumerate(keypoints):
        if is_zero(kp):
            continue

        x, y = int(kp["x"]), int(kp["y"])
        score = kp["score"]

        color = (0, 255, 0) if score >= CONF_THRESHOLD_HIGH else (0, 0, 255)

        cv2.circle(image, (x, y), 5, color, -1)

        cv2.putText(
            image,
            KEYPOINT_NAMES[idx],
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )


def is_zero(kp):
    return kp["x"] == 0 and kp["y"] == 0


# ============================================================
# 🔹 MAIN
# ============================================================

def main():
    label_data = load_label(FRAME_INDEX)
    frame = load_frame(VIDEO_PATH, FRAME_INDEX)

    print(f"Loaded frame {FRAME_INDEX}")
    print(f"Image size: {label_data['image_width']} x {label_data['image_height']}")
    print(f"Persons detected: {len(label_data['persons'])}")

    output = frame.copy()

    for idx, person in enumerate(label_data["persons"]):
        print(f"\n--- Person {idx + 1} ---")

        draw_bbox(output, person["bounding_box"])
        draw_skeleton(output, person["keypoints"])
        draw_keypoints(output, person["keypoints"])

    output_path = f"frame_{FRAME_INDEX}_keypoints.png"
    cv2.imwrite(output_path, output)

    print(f"\n✅ Saved to: {output_path}")

    Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB)).show()


if __name__ == "__main__":
    main()