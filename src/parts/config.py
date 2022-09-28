from enum import Enum


class DirName(Enum):
    TMP = "tmp"
    FRAMES = "01_frames"
    ALPHAPOSE = "02_alphapose"
    MEDIAPIPE = "03_mediapipe"
    POSETRIPLET = "04_posetriplet"


class FileName(Enum):
    ALPHAPOSE_RESULT = "alphapose-results.json"
    ALPHAPOSE_IMAGE = "alphapose.png"
    ALPHAPOSE_VIDEO = "alphapose.mp4"
