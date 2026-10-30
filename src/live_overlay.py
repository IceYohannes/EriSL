"""M0: live webcam preview with hand + upper-body landmark overlay.

Usage:  python -m src.live_overlay [--camera 0]
Keys:   q / Esc = quit

Nothing is recorded or saved. This only checks that MediaPipe runs and tracks well on your setup.
"""
import argparse
import time

import cv2
import numpy as np
from mediapipe.tasks.python.vision import HandLandmarksConnections

from src.landmarks import POSE_IDS, FrameLandmarks, LandmarkExtractor

HAND_COLORS = [(255, 128, 0), (0, 128, 255)]  # BGR: slot 0 ("Left") blue, slot 1 ("Right") orange
POSE_COLOR = (0, 220, 0)
# Upper-body skeleton drawn between our 7 kept pose points (indices into POSE_IDS order).
_P = list(POSE_IDS)
POSE_EDGES = [
    (_P.index("left_shoulder"), _P.index("right_shoulder")),
    (_P.index("left_shoulder"), _P.index("left_elbow")),
    (_P.index("left_elbow"), _P.index("left_wrist")),
    (_P.index("right_shoulder"), _P.index("right_elbow")),
    (_P.index("right_elbow"), _P.index("right_wrist")),
]


def to_px(points: np.ndarray, w: int, h: int) -> list[tuple[int, int]]:
    return [(int(x * w), int(y * h)) for x, y, _ in points]


def draw(frame: np.ndarray, lm: FrameLandmarks) -> None:
    h, w = frame.shape[:2]
    if lm.pose_found:
        pts = to_px(lm.pose, w, h)
        for a, b in POSE_EDGES:
            cv2.line(frame, pts[a], pts[b], POSE_COLOR, 2)
        for p in pts:
            cv2.circle(frame, p, 4, POSE_COLOR, -1)
    for slot in range(2):
        if not lm.presence[slot]:
            continue
        pts = to_px(lm.hands[slot], w, h)
        for c in HandLandmarksConnections.HAND_CONNECTIONS:
            cv2.line(frame, pts[c.start], pts[c.end], HAND_COLORS[slot], 2)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera", type=int, default=0)
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")

    start = time.monotonic()
    last_ts = -1
    fps = 0.0
    with LandmarkExtractor() as extractor:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            ts = int((time.monotonic() - start) * 1000)
            ts = max(ts, last_ts + 1)  # VIDEO mode needs strictly increasing timestamps
            t0 = time.monotonic()
            lm = extractor.process(frame, ts)
            fps = 0.9 * fps + 0.1 / max(time.monotonic() - t0, 1e-6)
            last_ts = ts

            # Inference ran on the un-mirrored frame; flip only the display so it feels like a mirror.
            draw(frame, lm)
            view = cv2.flip(frame, 1)
            status = (f"hands L:{int(lm.presence[0])} R:{int(lm.presence[1])}  "
                      f"pose:{int(lm.pose_found)}  {fps:4.1f} fps")
            cv2.putText(view, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.imshow("ErSL live landmarks (q to quit)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
