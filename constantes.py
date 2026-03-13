BODY_SKELETON = [
    (0, 1),          # shoulders
    (0, 2), (2, 4),  # left arm
    (1, 3), (3, 5),  # right arm
    (0, 6), (1, 7),  # torso
    (6, 7),          # hips
    (6, 8), (8, 10), # left leg
    (7, 9), (9, 11)  # right leg
]

# | New index | Original COCO | Joint          |
# | --------- | ------------- | -------------- |
# | 0         | 5             | left_shoulder  |
# | 1         | 6             | right_shoulder |
# | 2         | 7             | left_elbow     |
# | 3         | 8             | right_elbow    |
# | 4         | 9             | left_wrist     |
# | 5         | 10            | right_wrist    |
# | 6         | 11            | left_hip       |
# | 7         | 12            | right_hip      |
# | 8         | 13            | left_knee      |
# | 9         | 14            | right_knee     |
# | 10        | 15            | left_ankle     |
# | 11        | 16            | right_ankle    |


MAX_FRAMES = 2400
INDEX_LABEL_HIPS = 200
song = ""
SCORE_BUFFER_SIZE = 6 
# globals (initialized later)
song = None
VIDEO_PATH = None
AUDIO_PATH = None
LABEL_DIR = None


def set_song(new_song):
    global song, VIDEO_PATH, AUDIO_PATH, LABEL_DIR

    song = new_song
    VIDEO_PATH = f"musics/{song}/video/{song}.mp4"
    AUDIO_PATH = f"musics/{song}/audio/{song}_audio.mp3"
    LABEL_DIR = f"labels/{song}"
