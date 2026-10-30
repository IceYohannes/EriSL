import cv2
import numpy as np
import pytest

from src.import_videos import MAX_SIDE, find_videos, import_clip


def make_video(path, frames=20, size=(320, 240), fps=30.0):
    w, h = size
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for i in range(frames):
        writer.write(np.full((h, w, 3), i * 10 % 255, dtype=np.uint8))
    writer.release()


def frame_info(path):
    cap = cv2.VideoCapture(str(path))
    n, shape = 0, None
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n, shape = n + 1, frame.shape
    cap.release()
    return n, shape


def test_import_copies_all_frames(tmp_path):
    src = tmp_path / "phone.mp4"
    make_video(src, frames=20)
    dst = tmp_path / "out" / "s01" / "hello" / "001.mp4"
    n, fps = import_clip(src, dst)
    assert n == 20 and fps == pytest.approx(30.0)
    assert frame_info(dst) == (20, (240, 320, 3))


def test_import_downscales_large_video(tmp_path):
    src = tmp_path / "big.mp4"
    make_video(src, frames=3, size=(1920, 1080))
    dst = tmp_path / "out.mp4"
    import_clip(src, dst)
    _, shape = frame_info(dst)
    assert max(shape[:2]) == MAX_SIDE
    assert shape[:2] == (720, 1280)


def test_import_rejects_unreadable(tmp_path):
    bad = tmp_path / "not_a_video.mp4"
    bad.write_bytes(b"nope")
    with pytest.raises(IOError):
        import_clip(bad, tmp_path / "out.mp4")
    assert not (tmp_path / "out.mp4").exists()


def test_find_videos_expands_folders(tmp_path):
    (tmp_path / "b.MOV").write_bytes(b"")
    (tmp_path / "a.mp4").write_bytes(b"")
    (tmp_path / "notes.txt").write_text("x")
    other = tmp_path / "other"
    other.mkdir()
    (other / "c.mp4").write_bytes(b"")
    found = find_videos([tmp_path, other / "c.mp4"])
    assert [p.name for p in found] == ["a.mp4", "b.MOV", "c.mp4"]
