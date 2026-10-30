"""Synthetic end-to-end: fake extracted clips -> load_dataset -> splits -> baselines."""
import csv

import numpy as np
import pytest

from src.baseline import run
from src.dataset import load_dataset, signer_dependent_split, signer_independent_splits
from src.metadata import COLUMNS
from tests.test_features import make_frame

SIGNS = ["sign_a", "sign_b", "sign_c"]


def fake_clip(rng, sign_idx, n_frames, hand="right"):
    """Each sign moves the hands in a different direction; small per-clip noise."""
    hands, presence, pose = [], [], []
    for t in range(n_frames):
        h, p, po = make_frame(np.random.default_rng(0))  # same hand shape every frame
        direction = np.array([[1, 0], [0, 1], [-1, -1]][sign_idx], np.float32)
        h[..., :2] += direction * 0.01 * t + rng.normal(0, 0.003, h[..., :2].shape)
        hands.append(h); presence.append(p); pose.append(po)
    return dict(hands=np.stack(hands), presence=np.stack(presence), pose=np.stack(pose),
                pose_found=np.ones(n_frames, bool), fps=np.float32(30), dominant_hand=hand)


def build(tmp_path, signers=("s01", "s02"), reps=6, consent="true"):
    rng = np.random.default_rng(1)
    meta = tmp_path / "metadata.csv"
    processed = tmp_path / "processed"
    with open(meta, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS)
        w.writeheader()
        for s in signers:
            for k, sign in enumerate(SIGNS):
                for rep in range(1, reps + 1):
                    w.writerow({"signer_id": s, "sign_label": sign, "rep": rep, "device": "x",
                                "lighting": "x", "date": "2026-10-30", "consent_ok": consent,
                                "dominant_hand": "right"})
                    path = processed / s / sign / f"{rep:03d}.npz"
                    path.parent.mkdir(parents=True, exist_ok=True)
                    np.savez(path, **fake_clip(rng, k, int(rng.integers(20, 60))))
    return meta, processed


def test_load_shapes(tmp_path):
    ds = load_dataset(*build(tmp_path))
    assert ds.X.shape == (36, 30, 142)
    assert ds.labels == SIGNS
    assert sorted(set(ds.signers)) == ["s01", "s02"]


def test_no_consent_rows_ignored(tmp_path):
    meta, processed = build(tmp_path, signers=("s01",), consent="false")
    with pytest.raises(ValueError, match="no usable clips"):
        load_dataset(meta, processed)


def test_signer_independent_never_leaks(tmp_path):
    ds = load_dataset(*build(tmp_path, signers=("s01", "s02", "s03"), reps=3))
    folds = list(signer_independent_splits(ds))
    assert [f[0] for f in folds] == ["s01", "s02", "s03"]
    for signer, tr, te in folds:
        assert set(ds.signers[te]) == {signer}
        assert signer not in set(ds.signers[tr])


def test_signer_dependent_is_stratified_and_seeded(tmp_path):
    ds = load_dataset(*build(tmp_path))
    _, tr1, te1 = next(signer_dependent_split(ds))
    _, tr2, te2 = next(signer_dependent_split(ds))
    assert np.array_equal(te1, te2)
    assert set(ds.y[te1]) == set(range(len(SIGNS)))


def test_baselines_learn_easy_synthetic_signs(tmp_path):
    results = run(load_dataset(*build(tmp_path)))
    for name, res in results.items():
        assert res["signer_dependent"] > 0.9, name
        assert res["signer_independent"] > 0.9, name
        assert set(res["per_signer"]) == {"s01", "s02"}
