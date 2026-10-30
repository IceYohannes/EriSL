"""Download the MediaPipe Tasks model bundles into models/ (~13 MB total).

Usage:  python -m src.download_models
"""
from pathlib import Path
import urllib.request

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

MODEL_URLS = {
    "hand_landmarker.task": "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task",
    "pose_landmarker_lite.task": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
}


def main() -> None:
    MODELS_DIR.mkdir(exist_ok=True)
    for name, url in MODEL_URLS.items():
        dest = MODELS_DIR / name
        if dest.exists():
            print(f"[skip] {name} already present")
            continue
        print(f"[get ] {name}")
        urllib.request.urlretrieve(url, dest)
        print(f"       {dest.stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
