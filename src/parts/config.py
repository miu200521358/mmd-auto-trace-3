from enum import Enum


class DirName(Enum):
    TMP = "tmp"
    FRAMES = "01_frames"
    ALPHAPOSE = "02_alphapose"


class FileName(Enum):
    ALPHAPOSE_RESULT = "alphapose-results.json"
    ALPHAPOSE_IMAGE = "alphapose.png"
    ALPHAPOSE_VIDEO = "alphapose.mp4"
