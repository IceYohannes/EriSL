"""Record isolated-sign clips with a countdown, review, and metadata.

Usage:
    python -m src.collect --signer s01 --signs test_letter_1,test_letter_2 --reps 10 \
        --dominant-hand right --device laptop_webcam --lighting indoor_day

Per rep: SPACE starts a 3-2-1 countdown, then exactly 2 s are recorded. The clip then replays:
    SPACE = keep (saves .mp4 + metadata row)   r = redo   q = quit (current clip discarded)

Rules enforced (CLAUDE.md): sign labels must exist in docs/signs.md; you must confirm the signer's
informed consent at start, or nothing is saved. Clips are saved raw (no overlay drawn on them).
"""
import argparse
import datetime as dt
import time

import cv2
import numpy as np

from src.landmarks import LandmarkExtractor
from src.live_overlay import draw
from src.metadata import allowed_signs, append_metadata, clip_path, next_rep

COUNTDOWN_S = 3
CLIP_S = 2.0
WINDOW_NAME = "ErSL collect"
KEY_SPACE, KEY_ESC = 32, 27


def banner(img: np.ndarray, lines: list[str], color=(0, 255, 255)) -> np.ndarray:
    out = img.copy()
    for i, text in enumerate(lines):
        cv2.putText(out, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
        cv2.putText(out, text, (10, 30 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return out


class Recorder:
    def __init__(self, camera: int):
        self.cap = cv2.VideoCapture(camera)
        if not self.cap.isOpened():
            raise SystemExit(f"Could not open camera {camera}")
        self.extractor = LandmarkExtractor()
        self.t0 = time.monotonic()
        self.last_ts = -1

    def read(self) -> np.ndarray:
        ok, frame = self.cap.read()
        if not ok:
            raise SystemExit("Camera stopped delivering frames")
        return frame

    def preview(self, frame: np.ndarray) -> np.ndarray:
        """Frame with landmark overlay, mirrored for display."""
        ts = max(int((time.monotonic() - self.t0) * 1000), self.last_ts + 1)
        self.last_ts = ts
        view = frame.copy()
        draw(view, self.extractor.process(frame, ts))
        return cv2.flip(view, 1)

    def wait_for_start(self, lines: list[str]) -> bool:
        """Live preview until SPACE (True) or q/Esc (False)."""
        while True:
            cv2.imshow(WINDOW_NAME, banner(self.preview(self.read()), lines))
            key = cv2.waitKey(1) & 0xFF
            if key == KEY_SPACE:
                return True
            if key in (ord("q"), KEY_ESC):
                return False

    def countdown(self, sign: str) -> None:
        end = time.monotonic() + COUNTDOWN_S
        while (left := end - time.monotonic()) > 0:
            view = banner(self.preview(self.read()), [sign, f"{int(left) + 1}"], (0, 200, 255))
            cv2.imshow(WINDOW_NAME, view)
            cv2.waitKey(1)

    def capture(self, sign: str) -> tuple[list[np.ndarray], float]:
        """Record CLIP_S seconds. No landmark inference here, to keep the full camera frame rate."""
        frames = []
        start = time.monotonic()
        while time.monotonic() - start < CLIP_S:
            frame = self.read()
            frames.append(frame)
            cv2.imshow(WINDOW_NAME, banner(cv2.flip(frame, 1), [sign, "REC"], (0, 0, 255)))
            cv2.waitKey(1)
        fps = len(frames) / (time.monotonic() - start)
        return frames, fps

    def review(self, frames: list[np.ndarray], fps: float, title: str) -> str:
        """Loop the clip until the user decides. Returns 'keep', 'redo' or 'quit'."""
        delay = max(int(1000 / fps), 1)
        i = 0
        while True:
            lines = [title, "SPACE keep   r redo   q quit", f"{len(frames)} frames @ {fps:.1f} fps"]
            cv2.imshow(WINDOW_NAME, banner(cv2.flip(frames[i], 1), lines))
            key = cv2.waitKey(delay) & 0xFF
            if key == KEY_SPACE:
                return "keep"
            if key == ord("r"):
                return "redo"
            if key in (ord("q"), KEY_ESC):
                return "quit"
            i = (i + 1) % len(frames)

    def close(self) -> None:
        self.extractor.close()
        self.cap.release()
        cv2.destroyAllWindows()


def save_clip(frames: list[np.ndarray], fps: float, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    h, w = frames[0].shape[:2]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in frames:
        writer.write(f)
    writer.release()


def confirm_consent(signer: str) -> bool:
    answer = input(f"Has signer '{signer}' given informed consent to be recorded "
                   f"for this project? Type 'yes' to continue: ")
    return answer.strip().lower() == "yes"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--signer", required=True, help="anonymous id, e.g. s01 (never a real name)")
    p.add_argument("--signs", required=True, help="comma-separated labels from docs/signs.md")
    p.add_argument("--reps", type=int, default=10, help="reps to record per sign this session")
    p.add_argument("--dominant-hand", required=True, choices=["right", "left"])
    p.add_argument("--device", required=True, help="e.g. laptop_webcam, pixel7_front")
    p.add_argument("--lighting", required=True, help="e.g. indoor_day, indoor_lamp, outdoor")
    p.add_argument("--camera", type=int, default=0)
    args = p.parse_args()

    signs = [s.strip() for s in args.signs.split(",") if s.strip()]
    unknown = sorted(set(signs) - allowed_signs())
    if unknown:
        raise SystemExit(f"Not in docs/signs.md: {unknown}. Add (validated) signs there first.")
    if not confirm_consent(args.signer):
        raise SystemExit("No consent confirmed. Nothing recorded.")

    rec = Recorder(args.camera)
    try:
        for sign in signs:
            done = 0
            while done < args.reps:
                rep = next_rep(args.signer, sign)
                title = f"{sign}  rep {done + 1}/{args.reps}  (file {rep:03d})"
                if not rec.wait_for_start([title, "SPACE start   q quit"]):
                    return
                rec.countdown(sign)
                frames, fps = rec.capture(sign)
                decision = rec.review(frames, fps, title)
                if decision == "quit":
                    return
                if decision == "redo":
                    continue
                save_clip(frames, fps, clip_path(args.signer, sign, rep))
                append_metadata({
                    "signer_id": args.signer, "sign_label": sign, "rep": rep,
                    "device": args.device, "lighting": args.lighting,
                    "date": dt.date.today().isoformat(), "consent_ok": "true",
                    "dominant_hand": args.dominant_hand,
                })
                done += 1
                print(f"saved {sign} rep {rep:03d} ({len(frames)} frames @ {fps:.1f} fps)")
    finally:
        rec.close()


if __name__ == "__main__":
    main()
