"""End-to-end check on a synthetic (blank) video: runs real MediaPipe, so it needs models/."""
import cv2
import numpy as np
import pytest

from src.collect import save_clip
from src.extract import extract_all
from src.landmarks import HAND_MODEL, POSE_MODEL
from src.metadata import clip_path

pytestmark = pytest.mark.skipif(not (HAND_MODEL.exists() and POSE_MODEL.exists()),
                                reason="MediaPipe models not downloaded")


def meta(consent="true"):
    return {"signer_id": "s99", "sign_label": "test_letter_1", "rep": "1", "device": "synthetic",
            "lighting": "none", "date": "2026-10-30", "consent_ok": consent, "dominant_hand": "right"}


def test_blank_video_extracts_and_is_flagged(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    frames = [np.full((240, 320, 3), 40, np.uint8) for _ in range(12)]
    save_clip(frames, 30.0, clip_path("s99", "test_letter_1", 1, raw))

    report = extract_all([meta()], raw, out)

    data = np.load(out / "s99" / "test_letter_1" / "001.npz")
    assert data["hands"].shape == (12, 2, 21, 3)
    assert data["pose"].shape == (12, 7, 3)
    assert str(data["dominant_hand"]) == "right"
    assert report[0]["any_hand_rate"] == 0.0 and report[0]["flag"] == "LOW_HANDS"


def test_no_consent_is_skipped(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "processed"
    save_clip([np.zeros((64, 64, 3), np.uint8)] * 3, 30.0, clip_path("s99", "test_letter_1", 1, raw))
    assert extract_all([meta(consent="false")], raw, out) == []
    assert not out.exists()
