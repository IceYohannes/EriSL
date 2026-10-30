"""M1 step 4 sanity check: plot each sign's fingertip trajectories + run the baselines.

Usage:  python -m src.inspect_clips      (writes data/processed/inspect.png, gitignored)

What to look for: clips of the same sign should trace similar paths, different signs should
differ. A flat line at 0 means the hand wasn't detected (check lighting / framing).
"""
import matplotlib

matplotlib.use("Agg")  # file output only, no window needed
import matplotlib.pyplot as plt
import numpy as np

from src.baseline import main as run_baselines
from src.dataset import load_dataset
from src.metadata import PROCESSED_DIR

# After normalize_clip, the dominant hand is always slot 0 (see landmarks.py handedness note:
# MediaPipe labels a right-handed signer's right hand "Left" on un-mirrored frames, and
# left-handed signers are mirrored + slot-swapped).
INDEX_TIP = 8
X_COL, Y_COL = INDEX_TIP * 3, INDEX_TIP * 3 + 1


def plot(ds, out):
    fig, axes = plt.subplots(1, len(ds.labels), figsize=(4 * len(ds.labels), 4),
                             sharex=True, sharey=True, squeeze=False)
    for ax, (k, label) in zip(axes[0], enumerate(ds.labels)):
        for w in ds.X[ds.y == k]:
            present = w[:, 126] == 1
            ax.plot(w[present, X_COL], -w[present, Y_COL], alpha=0.5)  # -y: up is up
        ax.set_title(f"{label} ({(ds.y == k).sum()} clips)")
        ax.set_xlabel("x (shoulder widths)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("y (shoulder widths, up = +)")
    fig.suptitle("Dominant index fingertip, normalized (each line = one clip)")
    fig.tight_layout()
    fig.savefig(out, dpi=100)


def main() -> None:
    ds = load_dataset()
    out = PROCESSED_DIR / "inspect.png"
    plot(ds, out)
    print(f"trajectory plot: {out}\n")
    run_baselines()


if __name__ == "__main__":
    main()
