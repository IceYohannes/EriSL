import pytest

from src.metadata import COLUMNS, allowed_signs, append_metadata, clip_path, next_rep, read_metadata


def row(**overrides):
    base = {"signer_id": "s01", "sign_label": "hello", "rep": 1, "device": "webcam",
            "lighting": "day", "date": "2026-10-30", "consent_ok": "true", "dominant_hand": "right"}
    return {**base, **overrides}


def test_allowed_signs_reads_table_labels(tmp_path):
    md = tmp_path / "signs.md"
    md.write_text("# Signs\n| label | meaning |\n|---|---|\n| hello | hi |\n| `test_letter_1` | ? |\n"
                  "|  |  |\n| Bad Label | x |\n", encoding="utf-8")
    assert allowed_signs(md) == {"hello", "test_letter_1"}


def test_append_writes_header_once(tmp_path):
    csv_path = tmp_path / "metadata.csv"
    append_metadata(row(rep=1), csv_path)
    append_metadata(row(rep=2), csv_path)
    rows = read_metadata(csv_path)
    assert [r["rep"] for r in rows] == ["1", "2"]
    assert list(rows[0]) == COLUMNS


def test_append_refuses_without_consent(tmp_path):
    with pytest.raises(ValueError, match="consent"):
        append_metadata(row(consent_ok="false"), tmp_path / "m.csv")


def test_append_refuses_missing_column(tmp_path):
    bad = row()
    del bad["dominant_hand"]
    with pytest.raises(ValueError, match="dominant_hand"):
        append_metadata(bad, tmp_path / "m.csv")


def test_next_rep(tmp_path):
    assert next_rep("s01", "hello", tmp_path) == 1
    for rep in (1, 2, 5):
        p = clip_path("s01", "hello", rep, tmp_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.touch()
    assert next_rep("s01", "hello", tmp_path) == 6
    assert clip_path("s01", "hello", 6, tmp_path).name == "006.mp4"
