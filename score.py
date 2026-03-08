import numpy as np
from constantes import SCORE_BUFFER_SIZE
JOINT_WEIGHTS = np.array([
    1.2, 1.2,   # shoulders
    1.3, 1.3,   # elbows
    1.5, 1.5,   # wrists
    1.0, 1.0,   # hips
    1.2, 1.2,   # knees
    1.1, 1.1    # ankles
])


def get_score(prediction_key_points, labels_key_points):

    pred_xy, pred_conf = prediction_key_points[0]

    # remove head
    pred_xy = pred_xy[5:]
    pred_conf = pred_conf[5:]

    scores = []
    weights = []

    # torso scale
    shoulder_center = (pred_xy[0] + pred_xy[1]) / 2
    hip_center = (pred_xy[6] + pred_xy[7]) / 2
    torso = np.linalg.norm(shoulder_center - hip_center)

    sigma = torso * 0.25

    n = min(len(pred_xy), len(labels_key_points))

    for i in range(n):

        x_label, y_label, conf_label = labels_key_points[i]

        if pred_conf[i] < 0.3 or conf_label < 0.3:
            continue

        dist = np.linalg.norm(pred_xy[i] - np.array([x_label, y_label]))

        pose_score = np.exp(-(dist**2) / (2 * sigma**2))

        weight = pred_conf[i] * conf_label * JOINT_WEIGHTS[i]

        scores.append(pose_score * weight)
        weights.append(weight)

    if len(weights) == 0:
        return 0

    final_score = sum(scores) / sum(weights)

    final_score *= 100

    print("Score:", round(final_score, 1))

    return final_score


def get_smoothed_score(prediction_key_points, labels_key_points, score_buffer):
    score = get_score(prediction_key_points, labels_key_points)

    score_buffer.append(score)

    if len(score_buffer) > SCORE_BUFFER_SIZE:
        score_buffer.pop(0)

    smooth_score = sum(score_buffer) / len(score_buffer)

    print("Smooth score:", round(smooth_score, 1))
    return score_buffer