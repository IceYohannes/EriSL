"""Paths, metadata.csv and docs/signs.md helpers shared by collect.py and extract.py."""
import csv
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw" / "videos"
PROCESSED_DIR = ROOT / "data" / "processed"
METADATA_CSV = ROOT / "data" / "metadata.csv"
SIGNS_MD = ROOT / "docs" / "signs.md"

COLUMNS = ["signer_id", "sign_label", "rep", "device", "lighting", "date", "consent_ok", "dominant_hand"]
LABEL_RE = re.compile(r"^[a-z0-9_]+$")


def allowed_signs(signs_md: Path = SIGNS_MD) -> set[str]:
    """Every label in the first column of any markdown table in signs.md."""
    labels = set()
    for line in signs_md.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        first = line.strip("|").split("|")[0].strip().strip("`")
        if LABEL_RE.match(first) and first != "label":  # "label" is the header cell
            labels.add(first)
    return labels


def read_metadata(path: Path = METADATA_CSV) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def append_metadata(row: dict, path: Path = METADATA_CSV) -> None:
    missing = set(COLUMNS) - set(row)
    if missing:
        raise ValueError(f"metadata row missing {sorted(missing)}")
    if str(row["consent_ok"]).lower() != "true":
        raise ValueError("refusing to record a clip without consent_ok=true")
    new_file = not path.exists() or path.stat().st_size == 0
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerow({c: row[c] for c in COLUMNS})


def clip_path(signer_id: str, sign_label: str, rep: int, root: Path = RAW_DIR) -> Path:
    return root / signer_id / sign_label / f"{rep:03d}.mp4"


def next_rep(signer_id: str, sign_label: str, root: Path = RAW_DIR) -> int:
    folder = root / signer_id / sign_label
    reps = [int(p.stem) for p in folder.glob("*.mp4") if p.stem.isdigit()] if folder.exists() else []
    return max(reps, default=0) + 1
