from enum import Enum


class DirName(Enum):
    TMP = "tmp"
    FRAMES = "01_frames"
    ALPHAPOSE = "02_alphapose"


class FileName(Enum):
    ORIGINAL = "original.mp4"
    ALPHAPOSE_RESULT = "alphapose-results.json"
    ALPHAPOSE_VIDEO = "alphapose.mp4"
