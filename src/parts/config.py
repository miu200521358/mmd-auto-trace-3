from enum import Enum


class DirName(Enum):
    TMP = "tmp"
    FRAMES = "01_frames"
    ALPHAPOSE = "02_alphapose"
    MULTIPOSE = "03_multipose"
    POSETRIPLET = "04_posetriplet"
    MIX = "05_mix"
    MOTION = "06_motion"


class FileName(Enum):
    ALPHAPOSE_RESULT = "alphapose-results.json"
    ALPHAPOSE_VIDEO = "alphapose.mp4"


SMPL_JOINT_29 = {
    "Pelvis": 0,
    "LHip": 1,
    "RHip": 2,
    "Spine1": 3,
    "LKnee": 4,
    "RKnee": 5,
    "Spine2": 6,
    "LAnkle": 7,
    "RAnkle": 8,
    "Spine3": 9,
    "LFoot": 10,
    "RFoot": 11,
    "Neck": 12,
    "LCollar": 13,
    "RCollar": 14,
    "Jaw": 15,
    "LShoulder": 16,
    "RShoulder": 17,
    "LElbow": 18,
    "RElbow": 19,
    "LWrist": 20,
    "RWrist": 21,
    "LThumb": 22,
    "RThumb": 23,
    "Head": 24,
    "LMiddle": 25,
    "RMiddle": 26,
    "LBigtoe": 27,
    "RBigtoe": 28,
}

SMPL_JOINT_24 = {
    "Pelvis": 0,
    "LHip": 1,
    "RHip": 2,
    "Spine1": 3,
    "LKnee": 4,
    "RKnee": 5,
    "Spine2": 6,
    "LAnkle": 7,
    "RAnkle": 8,
    "Spine3": 9,
    "LFoot": 10,
    "RFoot": 11,
    "Head": 12,
    "LCollar": 13,
    "RCollar": 14,
    "Nose": 15,
    "LShoulder": 16,
    "RShoulder": 17,
    "LElbow": 18,
    "RElbow": 19,
    "LWrist": 20,
    "RWrist": 21,
    "LMiddle": 22,
    "RMiddle": 23,
}
