"""Normalization + windowing: raw landmark arrays -> model input.

THIS IS THE CONTRACT WITH THE ANDROID APP. Any change here must be mirrored in Kotlin and the
golden test file (tests/golden_features.npz) regenerated. See docs/PLAN.md, M1 step 1.

Frame vector layout (FRAME_DIM = 142 floats):
    [0:63]    hand slot 0 ("Left" per MediaPipe), 21 points x (x, y, z)
    [63:126]  hand slot 1 ("Right" per MediaPipe), 21 points x (x, y, z)
    [126:128] presence flags for slot 0, slot 1
    [128:142] 7 pose points (POSE_IDS order) x (x, y); pose z is dropped as too noisy

Normalization:
    origin = shoulder midpoint, scale = shoulder width (in x,y image units)
    x, y -> (v - origin) / scale;  hand z -> z / scale (it's already relative to the wrist)
    missing hands stay all zeros
Left-handed signers: x -> -x and swap hand slots, so every signer looks right-dominant.
"""
import numpy as np

from src.landmarks import POSE_IDS

NUM_POSE = len(POSE_IDS)
FRAME_DIM = 2 * 21 * 3 + 2 + NUM_POSE * 2  # 142
WINDOW = 30

_L_SHOULDER = list(POSE_IDS).index("left_shoulder")
_R_SHOULDER = list(POSE_IDS).index("right_shoulder")
_MIN_SCALE = 1e-3  # guards against a degenerate pose (shoulders on top of each other)


def body_frame(pose: np.ndarray) -> tuple[np.ndarray, float] | None:
    """Origin (x, y) and scale from the shoulders, or None if the pose is unusable."""
    left, right = pose[_L_SHOULDER, :2], pose[_R_SHOULDER, :2]
    scale = float(np.linalg.norm(left - right))
    if scale < _MIN_SCALE:
        return None
    return (left + right) / 2, scale


def normalize_frame(hands: np.ndarray, presence: np.ndarray, pose: np.ndarray,
                    origin: np.ndarray, scale: float, left_handed: bool = False) -> np.ndarray:
    """One frame of raw landmarks -> (FRAME_DIM,) float32 vector."""
    hands = hands.astype(np.float32).copy()
    presence = presence.astype(np.float32).copy()
    pose_xy = pose[:, :2].astype(np.float32).copy()

    hands[..., :2] = (hands[..., :2] - origin) / scale
    hands[..., 2] /= scale
    hands[presence == 0] = 0.0  # missing hand: exact zeros, not "-origin/scale"
    pose_xy = (pose_xy - origin) / scale

    if left_handed:
        hands[..., 0] *= -1
        pose_xy[:, 0] *= -1
        hands = hands[::-1]
        presence = presence[::-1]
        # Swap left/right body points too, so "left_shoulder" stays on the same side of the image.
        pose_xy = pose_xy[_mirror_pose_order()]

    return np.concatenate([hands.reshape(-1), presence, pose_xy.reshape(-1)]).astype(np.float32)


def _mirror_pose_order() -> list[int]:
    names = list(POSE_IDS)
    swapped = [n.replace("left", "@").replace("right", "left").replace("@", "right") for n in names]
    return [names.index(n) for n in swapped]


def normalize_clip(hands: np.ndarray, presence: np.ndarray, pose: np.ndarray,
                   pose_found: np.ndarray, left_handed: bool = False) -> np.ndarray:
    """A clip of T raw frames -> (T', FRAME_DIM). Frames before the first usable pose are dropped;
    after that, frames without a usable pose reuse the last good pose (and its origin/scale)."""
    out = []
    ref = None
    last_pose = None
    for t in range(len(hands)):
        if pose_found[t] and (frame_ref := body_frame(pose[t])) is not None:
            ref, last_pose = frame_ref, pose[t]
        if ref is None:
            continue
        out.append(normalize_frame(hands[t], presence[t], last_pose, ref[0], ref[1], left_handed))
    return np.stack(out) if out else np.zeros((0, FRAME_DIM), np.float32)


def to_window(frames: np.ndarray, length: int = WINDOW) -> np.ndarray:
    """Resample (T, D) -> (length, D) by linear interpolation over time.
    Presence flags are snapped back to 0/1 so they stay flags."""
    if len(frames) == 0:
        raise ValueError("empty clip")
    if len(frames) == 1:
        out = np.repeat(frames, length, axis=0)
    else:
        src = np.linspace(0, 1, len(frames))
        dst = np.linspace(0, 1, length)
        out = np.stack([np.interp(dst, src, frames[:, d]) for d in range(frames.shape[1])], axis=1)
    out[:, 126:128] = np.round(out[:, 126:128])
    for slot in range(2):  # a hand interpolated against a missing frame is half-garbage: drop it
        out[out[:, 126 + slot] == 0, slot * 63:(slot + 1) * 63] = 0.0
    return out.astype(np.float32)
