from pathlib import Path

import numpy as np
import pytest

from src.features import FRAME_DIM, WINDOW, normalize_clip, normalize_frame, body_frame, to_window
from src.landmarks import POSE_IDS

GOLDEN = Path(__file__).parent / "golden_features.npz"
P = list(POSE_IDS)


def make_frame(rng, shift=(0.0, 0.0), zoom=1.0, presence=(1, 1)):
    """A plausible raw frame: shoulders at y=0.6, 0.3 apart, hands somewhere above."""
    pose = np.zeros((len(P), 3), np.float32)
    pose[P.index("left_shoulder")] = (0.65, 0.6, 0)
    pose[P.index("right_shoulder")] = (0.35, 0.6, 0)
    pose[P.index("nose")] = (0.5, 0.4, 0)
    pose[P.index("left_elbow")] = (0.7, 0.8, 0)
    pose[P.index("right_elbow")] = (0.3, 0.8, 0)
    pose[P.index("left_wrist")] = (0.6, 0.5, 0)
    pose[P.index("right_wrist")] = (0.4, 0.5, 0)
    hands = rng.uniform(0.3, 0.7, (2, 21, 3)).astype(np.float32)
    hands[..., 2] = rng.uniform(-0.05, 0.05, (2, 21))
    presence = np.array(presence, np.float32)
    hands[presence == 0] = 0

    center = np.array([0.5, 0.6], np.float32)
    for arr in (pose, hands):
        arr[..., :2] = (arr[..., :2] - center) * zoom + center + np.array(shift, np.float32)
    hands[..., 2] *= zoom
    hands[presence == 0] = 0
    return hands, presence, pose


def norm(hands, presence, pose, left_handed=False):
    origin, scale = body_frame(pose)
    return normalize_frame(hands, presence, pose, origin, scale, left_handed)


def test_frame_dim():
    rng = np.random.default_rng(0)
    assert norm(*make_frame(rng)).shape == (FRAME_DIM,)


@pytest.mark.parametrize("shift,zoom", [((0.1, -0.05), 1.0), ((0, 0), 0.5), ((-0.1, 0.1), 1.3)])
def test_invariant_to_position_and_distance(shift, zoom):
    base = norm(*make_frame(np.random.default_rng(1)))
    moved = norm(*make_frame(np.random.default_rng(1), shift=shift, zoom=zoom))
    assert np.allclose(base, moved, atol=1e-5)


def test_shoulders_land_on_fixed_points():
    vec = norm(*make_frame(np.random.default_rng(2)))
    pose_xy = vec[128:].reshape(-1, 2)
    assert np.allclose(pose_xy[P.index("left_shoulder")], (0.5, 0.0))
    assert np.allclose(pose_xy[P.index("right_shoulder")], (-0.5, 0.0))


def test_missing_hand_is_exact_zeros():
    vec = norm(*make_frame(np.random.default_rng(3), presence=(0, 1)))
    assert not vec[0:63].any()
    assert vec[63:126].any()
    assert vec[126:128].tolist() == [0.0, 1.0]


def test_left_handed_mirror():
    hands, presence, pose = make_frame(np.random.default_rng(4), presence=(1, 0))
    right = norm(hands, presence, pose)
    left = norm(hands, presence, pose, left_handed=True)
    # The detected hand moves to the other slot, with x negated.
    assert left[126:128].tolist() == [0.0, 1.0]
    r = right[0:63].reshape(21, 3)
    l = left[63:126].reshape(21, 3)
    assert np.allclose(l[:, 0], -r[:, 0]) and np.allclose(l[:, 1:], r[:, 1:])
    # Mirroring a symmetric body is a no-op on pose.
    assert np.allclose(left[128:], right[128:], atol=1e-6)


def test_degenerate_pose_rejected():
    pose = np.zeros((len(P), 3), np.float32)
    assert body_frame(pose) is None


def test_clip_skips_until_pose_then_carries_forward():
    rng = np.random.default_rng(5)
    frames = [make_frame(rng) for _ in range(4)]
    hands = np.stack([f[0] for f in frames])
    presence = np.stack([f[1] for f in frames])
    pose = np.stack([f[2] for f in frames])
    pose_found = np.array([False, True, False, True])
    pose[2] = 0  # no person detected in frame 2
    out = normalize_clip(hands, presence, pose, pose_found)
    assert out.shape == (3, FRAME_DIM)
    assert np.allclose(out[1, 128:], out[0, 128:])  # frame 2 reused frame 1's pose


@pytest.mark.parametrize("length", [1, 12, 30, 75])
def test_window_shape(length):
    frames = np.random.default_rng(6).normal(size=(length, FRAME_DIM)).astype(np.float32)
    frames[:, 126:128] = 1
    assert to_window(frames).shape == (WINDOW, FRAME_DIM)


def test_window_keeps_endpoints_and_flags():
    frames = np.zeros((10, FRAME_DIM), np.float32)
    frames[:, 0] = np.arange(10)
    frames[:, 126] = 1
    frames[5:, 127] = 1
    frames[5:, 63:126] = 2.0
    w = to_window(frames)
    assert w[0, 0] == 0 and w[-1, 0] == 9
    assert set(np.unique(w[:, 126:128])) <= {0.0, 1.0}
    assert not w[w[:, 127] == 0, 63:126].any()  # hand zeroed wherever its flag is 0


def test_golden_file_matches():
    """Locks the exact numbers. The Kotlin port must reproduce `expected` from `inputs`.
    Regenerate only on a deliberate format change: python -m tests.make_golden"""
    g = np.load(GOLDEN)
    clip = normalize_clip(g["hands"], g["presence"], g["pose"], g["pose_found"], bool(g["left_handed"]))
    assert np.allclose(to_window(clip), g["expected"], atol=1e-6)
