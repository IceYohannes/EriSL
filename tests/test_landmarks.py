from types import SimpleNamespace as NS

import numpy as np

from src.landmarks import POSE_IDS, to_arrays


def _hand(value: float):
    return [NS(x=value, y=value, z=value) for _ in range(21)]


def _label(name: str):
    return [NS(category_name=name)]


def _pose():
    return [NS(x=i / 100, y=i / 100, z=0.0) for i in range(33)]


def test_missing_everything_gives_zeros():
    lm = to_arrays(NS(hand_landmarks=[], handedness=[]), NS(pose_landmarks=[]))
    assert lm.hands.shape == (2, 21, 3) and not lm.hands.any()
    assert lm.presence.tolist() == [0.0, 0.0]
    assert lm.pose.shape == (len(POSE_IDS), 3) and not lm.pose.any()
    assert lm.pose_found is False


def test_hands_go_to_slots_by_label():
    hands = NS(hand_landmarks=[_hand(0.7), _hand(0.3)], handedness=[_label("Right"), _label("Left")])
    lm = to_arrays(hands, NS(pose_landmarks=[]))
    assert lm.presence.tolist() == [1.0, 1.0]
    assert np.allclose(lm.hands[0], 0.3)  # "Left"
    assert np.allclose(lm.hands[1], 0.7)  # "Right"


def test_duplicate_label_keeps_first():
    hands = NS(hand_landmarks=[_hand(0.1), _hand(0.9)], handedness=[_label("Left"), _label("Left")])
    lm = to_arrays(hands, NS(pose_landmarks=[]))
    assert lm.presence.tolist() == [1.0, 0.0]
    assert np.allclose(lm.hands[0], 0.1)


def test_pose_picks_upper_body_indices():
    lm = to_arrays(NS(hand_landmarks=[], handedness=[]), NS(pose_landmarks=[_pose()]))
    assert lm.pose_found
    assert np.allclose(lm.pose[:, 0], [i / 100 for i in POSE_IDS.values()])
