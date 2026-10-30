"""Import videos recorded elsewhere (e.g. on a phone) into the dataset.

Usage:
    python -m src.import_videos --signer s01 --sign test_letter_1 --dominant-hand right \
        --device pixel7_back --lighting indoor_day  C:/Users/me/Videos/letter1/

Each video file = ONE rep of ONE sign. Pass files and/or folders (folders are scanned for video
files, sorted by name). Every clip is re-encoded to .mp4 (long side at most 1280 px, which is
plenty for MediaPipe and keeps files small), saved to data/raw/videos/<signer>/<sign>/<rep>.mp4,
and gets a row in data/metadata.csv. Your original files are never modified or deleted.

Tips for phone recordings: one sign per clip, ~2 s, start and end with hands down,
upper body (head to waist) in frame. Clips longer than 4 s get a warning, because extra idle
frames dilute the 30-frame window that features.py resamples every clip to.

Same rules as collect.py: the sign must exist in docs/signs.md and you must confirm consent.
"""
import argparse
import datetime as dt
from pathlib import Path

import cv2

from src.metadata import allowed_signs, append_metadata, clip_path, confirm_consent, next_rep

VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp", ".webm"}
MAX_SIDE = 1280
MIN_S, MAX_S = 0.5, 4.0


def find_videos(paths: list[Path]) -> list[Path]:
    """Expand folders into their video files; keep explicit files as given."""
    found = []
    for p in paths:
        if p.is_dir():
            found += sorted(f for f in p.iterdir() if f.suffix.lower() in VIDEO_EXTS)
        elif p.is_file():
            found.append(p)
        else:
            raise SystemExit(f"Not found: {p}")
    return found


def import_clip(src: Path, dst: Path) -> tuple[int, float]:
    """Re-encode src to an .mp4 at dst, downscaled if large. Returns (frames, fps)."""
    cap = cv2.VideoCapture(str(src))  # OpenCV applies the phone's rotation metadata itself
    if not cap.isOpened():
        raise IOError(f"cannot open {src}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    writer, n = None, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            h, w = frame.shape[:2]
            scale = MAX_SIDE / max(h, w)
            if scale < 1:
                frame = cv2.resize(frame, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
            if writer is None:
                dst.parent.mkdir(parents=True, exist_ok=True)
                size = (frame.shape[1], frame.shape[0])
                writer = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
            writer.write(frame)
            n += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    if n == 0:
        raise IOError(f"no frames in {src}")
    return n, fps


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="+", type=Path, help="video files and/or folders of videos")
    p.add_argument("--signer", required=True, help="anonymous id, e.g. s01 (never a real name)")
    p.add_argument("--sign", required=True, help="one label from docs/signs.md")
    p.add_argument("--dominant-hand", required=True, choices=["right", "left"])
    p.add_argument("--device", required=True, help="e.g. pixel7_back, iphone12_front")
    p.add_argument("--lighting", required=True, help="e.g. indoor_day, indoor_lamp, outdoor")
    args = p.parse_args()

    if args.sign not in allowed_signs():
        raise SystemExit(f"'{args.sign}' is not in docs/signs.md. Add (validated) signs there first.")
    videos = find_videos(args.paths)
    if not videos:
        raise SystemExit("No video files found.")
    print(f"{len(videos)} video(s) -> {args.signer}/{args.sign}")
    if not confirm_consent(args.signer):
        raise SystemExit("No consent confirmed. Nothing imported.")

    for src in videos:
        rep = next_rep(args.signer, args.sign)
        dst = clip_path(args.signer, args.sign, rep)
        try:
            n, fps = import_clip(src, dst)
        except IOError as e:
            print(f"[fail] {src.name}: {e}")
            continue
        recorded = dt.date.fromtimestamp(src.stat().st_mtime).isoformat()
        append_metadata({
            "signer_id": args.signer, "sign_label": args.sign, "rep": rep,
            "device": args.device, "lighting": args.lighting,
            "date": recorded, "consent_ok": "true", "dominant_hand": args.dominant_hand,
        })
        secs = n / fps
        warn = "  (long: trim to ~2 s?)" if secs > MAX_S else "  (very short?)" if secs < MIN_S else ""
        print(f"[ok  ] {src.name} -> rep {rep:03d}  {n} frames, {secs:.1f} s{warn}")
    print("\nNext: python -m src.extract")


if __name__ == "__main__":
    main()
