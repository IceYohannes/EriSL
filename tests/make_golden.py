"""Regenerate tests/golden_features.npz. Only run on a deliberate feature-format change,
and update the Kotlin port at the same time.

Usage:  python -m tests.make_golden
"""
from pathlib import Path

import numpy as np

from src.features import normalize_clip, to_window
from tests.test_features import make_frame

OUT = Path(__file__).parent / "golden_features.npz"


def main() -> None:
    rng = np.random.default_rng(42)
    presences = [(1, 1)] * 10 + [(0, 1)] * 5 + [(1, 0)] * 5 + [(1, 1)] * 20
    frames = [make_frame(rng, shift=(0.01 * i, 0), zoom=1 + 0.01 * i, presence=p)
              for i, p in enumerate(presences)]
    hands = np.stack([f[0] for f in frames])
    presence = np.stack([f[1] for f in frames])
    pose = np.stack([f[2] for f in frames])
    pose_found = np.ones(len(frames), bool)
    pose_found[0] = pose_found[25] = False
    left_handed = False

    expected = to_window(normalize_clip(hands, presence, pose, pose_found, left_handed))
    np.savez(OUT, hands=hands, presence=presence, pose=pose, pose_found=pose_found,
             left_handed=left_handed, expected=expected)
    print(f"wrote {OUT}  expected shape {expected.shape}")


if __name__ == "__main__":
    main()
