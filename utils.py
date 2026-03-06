import os
import json
import numpy as np
import pygame
from constantes import INDEX_LABEL_HIPS

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

def get_audio(audio_path:str):

    pygame.mixer.init()
    pygame.mixer.music.load(audio_path)
    pygame.mixer.music.play()

def get_labels(labels_dir="labels"):
    labels = {}
    for f in os.listdir(labels_dir):
        with open(os.path.join(labels_dir, f)) as file:
            labels[int(f.split(".")[0])] = json.load(file)
    return labels
if __name__ == "__main__":
    print(get_label_torso())