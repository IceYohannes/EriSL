"""MediaPipe Tasks wrapper: one video frame -> raw (un-normalized) landmark arrays.

Used by the live overlay, collect.py and extract.py so every tool sees landmarks the same way.

Raw per-frame layout (normalization happens later, in features.py):
    hands    (2, 21, 3)  index 0 = left hand, 1 = right hand; zeros when missing
    presence (2,)        1.0 if that hand was detected, else 0.0
    pose     (7, 3)      POSE_IDS below; zeros when no person detected

Handedness convention: we always run on the *un-mirrored* camera frame and store hands by the
label MediaPipe reports. MediaPipe assumes a mirrored (selfie) image, so on an un-mirrored frame
its "Left" is the signer's right hand. That is fine as long as Android does exactly the same:
feed un-mirrored frames, keep MediaPipe's label. Do not "fix" the swap in one place only.
"""
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
HAND_MODEL = MODELS_DIR / "hand_landmarker.task"
POSE_MODEL = MODELS_DIR / "pose_landmarker_lite.task"

# Upper-body pose points we keep (MediaPipe Pose indices).
POSE_IDS = {
    "nose": 0,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
}
NUM_HAND_POINTS = 21


@dataclass
class FrameLandmarks:
    hands: np.ndarray     # (2, 21, 3) float32
    presence: np.ndarray  # (2,) float32
    pose: np.ndarray      # (7, 3) float32
    pose_found: bool


class LandmarkExtractor:
    """Runs HandLandmarker + PoseLandmarker in VIDEO mode (uses tracking between frames).

    Use as a context manager. Timestamps passed to `process` must strictly increase.
    """

    def __init__(self, num_hands: int = 2, min_confidence: float = 0.5):
        for path in (HAND_MODEL, POSE_MODEL):
            if not path.exists():
                raise FileNotFoundError(f"{path} missing. Run: python -m src.download_models")
        self._hand = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(HAND_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=num_hands,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        ))
        self._pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(POSE_MODEL)),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=min_confidence,
            min_pose_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
        ))
        self.last_hand_result = None
        self.last_pose_result = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self) -> None:
        self._hand.close()
        self._pose.close()

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int) -> FrameLandmarks:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        self.last_hand_result = self._hand.detect_for_video(image, timestamp_ms)
        self.last_pose_result = self._pose.detect_for_video(image, timestamp_ms)
        return to_arrays(self.last_hand_result, self.last_pose_result)


def to_arrays(hand_result, pose_result) -> FrameLandmarks:
    hands = np.zeros((2, NUM_HAND_POINTS, 3), dtype=np.float32)
    presence = np.zeros(2, dtype=np.float32)
    for landmarks, handedness in zip(hand_result.hand_landmarks, hand_result.handedness):
        slot = 0 if handedness[0].category_name == "Left" else 1
        if presence[slot]:  # two hands got the same label; keep the first (higher score)
            continue
        hands[slot] = [(p.x, p.y, p.z) for p in landmarks]
        presence[slot] = 1.0

    pose = np.zeros((len(POSE_IDS), 3), dtype=np.float32)
    pose_found = bool(pose_result.pose_landmarks)
    if pose_found:
        person = pose_result.pose_landmarks[0]
        pose[:] = [(person[i].x, person[i].y, person[i].z) for i in POSE_IDS.values()]
    return FrameLandmarks(hands, presence, pose, pose_found)
