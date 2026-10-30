"""Videos -> raw landmark sequences (.npz) + a quality report.

Usage:  python -m src.extract [--force]

Only clips listed in data/metadata.csv with consent_ok=true are processed. Output per clip:
    data/processed/<signer>/<sign>/<rep>.npz with
        hands (T,2,21,3), presence (T,2), pose (T,7,3), pose_found (T,), fps, dominant_hand
Arrays are raw MediaPipe output; normalization happens later in features.py.
Quality report: data/processed/quality.csv (clips with < 80 % hand detection are flagged).
"""
import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from src.landmarks import LandmarkExtractor
from src.metadata import PROCESSED_DIR, RAW_DIR, clip_path, read_metadata

MIN_HAND_RATE = 0.8


def extract_clip(video: Path) -> dict:
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise IOError(f"cannot open {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    hands, presence, pose, pose_found = [], [], [], []
    # A fresh extractor per clip: VIDEO-mode tracking must not carry over between clips.
    with LandmarkExtractor() as extractor:
        i = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            lm = extractor.process(frame, int(i * 1000 / fps))
            hands.append(lm.hands)
            presence.append(lm.presence)
            pose.append(lm.pose)
            pose_found.append(lm.pose_found)
            i += 1
    cap.release()
    if not hands:
        raise IOError(f"no frames in {video}")
    return {
        "hands": np.stack(hands), "presence": np.stack(presence),
        "pose": np.stack(pose), "pose_found": np.array(pose_found), "fps": np.float32(fps),
    }


def quality(data: dict) -> dict:
    any_hand = data["presence"].max(axis=1)
    return {
        "frames": len(any_hand),
        "pose_rate": round(float(data["pose_found"].mean()), 3),
        "left_slot_rate": round(float(data["presence"][:, 0].mean()), 3),
        "right_slot_rate": round(float(data["presence"][:, 1].mean()), 3),
        "any_hand_rate": round(float(any_hand.mean()), 3),
    }


def extract_all(rows: list[dict], raw_root: Path, out_root: Path, force: bool = False) -> list[dict]:
    report = []
    for r in rows:
        if r["consent_ok"].lower() != "true":
            print(f"[skip] no consent: {r['signer_id']}/{r['sign_label']}/{r['rep']}")
            continue
        rep = int(r["rep"])
        src = clip_path(r["signer_id"], r["sign_label"], rep, raw_root)
        dst = out_root / r["signer_id"] / r["sign_label"] / f"{rep:03d}.npz"
        name = f"{r['signer_id']}/{r['sign_label']}/{rep:03d}"
        if not src.exists():
            print(f"[miss] {name}: video not found")
            continue
        if dst.exists() and not force:
            data = dict(np.load(dst))
        else:
            data = extract_clip(src)
            dst.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(dst, dominant_hand=r["dominant_hand"], **data)
        q = {"clip": name, **quality(data)}
        q["flag"] = "LOW_HANDS" if q["any_hand_rate"] < MIN_HAND_RATE else ""
        report.append(q)
        print(f"[ok  ] {name}  {q['frames']} frames  hands {q['any_hand_rate']:.0%}  "
              f"pose {q['pose_rate']:.0%}  {q['flag']}")
    return report


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--force", action="store_true", help="re-extract clips that already have a .npz")
    args = p.parse_args()

    report = extract_all(read_metadata(), RAW_DIR, PROCESSED_DIR, args.force)
    if not report:
        print("Nothing extracted. Record clips with src.collect first.")
        return
    out = PROCESSED_DIR / "quality.csv"
    with open(out, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(report[0]))
        writer.writeheader()
        writer.writerows(report)
    flagged = [q["clip"] for q in report if q["flag"]]
    print(f"\n{len(report)} clips, {len(flagged)} flagged. Report: {out}")
    for c in flagged:
        print(f"  re-record? {c}")


if __name__ == "__main__":
    main()
