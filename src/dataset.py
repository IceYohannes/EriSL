"""Load extracted clips as fixed-size windows, and make the two kinds of train/test split.

    X        (N, 30, 142) float32   model input, see features.py
    y        (N,) int               index into `labels`
    signers  (N,) str               who signed it (used for splitting, never as a feature)

Splits (CLAUDE.md rule 1, always report both):
    signer-dependent:   random stratified split of clips; same people in train and test
    signer-independent: leave-one-signer-out; the test person is never seen in training
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split

from src.features import normalize_clip, to_window
from src.metadata import METADATA_CSV, PROCESSED_DIR, read_metadata

SEED = 42


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    signers: np.ndarray
    labels: list[str]
    clips: list[str]


def load_dataset(metadata_csv: Path = METADATA_CSV, processed: Path = PROCESSED_DIR,
                 include_test_signs: bool = True) -> Dataset:
    rows = [r for r in read_metadata(metadata_csv) if r["consent_ok"].lower() == "true"]
    if not include_test_signs:
        rows = [r for r in rows if not r["sign_label"].startswith("test_")]
    labels = sorted({r["sign_label"] for r in rows})
    X, y, signers, clips = [], [], [], []
    for r in rows:
        name = f"{r['signer_id']}/{r['sign_label']}/{int(r['rep']):03d}"
        path = processed / f"{name}.npz"
        if not path.exists():
            print(f"[miss] {name}: not extracted (run src.extract)")
            continue
        d = np.load(path)
        frames = normalize_clip(d["hands"], d["presence"], d["pose"], d["pose_found"],
                                left_handed=str(d["dominant_hand"]) == "left")
        if len(frames) == 0:
            print(f"[skip] {name}: no usable pose in any frame")
            continue
        X.append(to_window(frames))
        y.append(labels.index(r["sign_label"]))
        signers.append(r["signer_id"])
        clips.append(name)
    if not X:
        raise ValueError("no usable clips found")
    return Dataset(np.stack(X), np.array(y), np.array(signers), labels, clips)


def signer_dependent_split(ds: Dataset, test_size: float = 0.25):
    """Yields one (train_idx, test_idx) pair: random but stratified by sign."""
    idx = np.arange(len(ds.y))
    train, test = train_test_split(idx, test_size=test_size, stratify=ds.y, random_state=SEED)
    yield "random", train, test


def signer_independent_splits(ds: Dataset):
    """Yields (held_out_signer, train_idx, test_idx), once per signer. Needs 2+ signers."""
    for s in sorted(set(ds.signers)):
        test = np.flatnonzero(ds.signers == s)
        train = np.flatnonzero(ds.signers != s)
        yield s, train, test
