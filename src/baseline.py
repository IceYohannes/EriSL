"""Baseline classifiers on flattened windows: kNN and a small MLP (scikit-learn).

Usage:  python -m src.baseline [--no-test-signs]

Why baselines first: if a 1-nearest-neighbour lookup already gets 95 %, the GRU must beat
that to be worth it. They also catch pipeline bugs: near-chance accuracy with clean data
means something upstream (labels, normalization) is broken.
"""
import argparse

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.dataset import SEED, Dataset, load_dataset, signer_dependent_split, signer_independent_splits


def models() -> dict:
    return {
        "knn1": make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=1)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128,), max_iter=500,
                                                             early_stopping=False, random_state=SEED)),
    }


def run(ds: Dataset) -> dict:
    """Returns {model: {"signer_dependent": acc, "signer_independent": acc or None, "per_signer": {...}}}."""
    X = ds.X.reshape(len(ds.X), -1)
    results = {}

    def score(name, tr, te):  # fresh, untrained model for every split
        model = models()[name].fit(X[tr], ds.y[tr])
        return accuracy_score(ds.y[te], model.predict(X[te]))

    for name in models():
        res = {}
        _, tr, te = next(signer_dependent_split(ds))
        res["signer_dependent"] = score(name, tr, te)

        per_signer = {}
        if len(set(ds.signers)) >= 2:
            for signer, tr, te in signer_independent_splits(ds):
                per_signer[signer] = score(name, tr, te)
        res["per_signer"] = per_signer
        res["signer_independent"] = float(np.mean(list(per_signer.values()))) if per_signer else None
        results[name] = res
    return results


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--no-test-signs", action="store_true", help="exclude test_* signs")
    args = p.parse_args()

    ds = load_dataset(include_test_signs=not args.no_test_signs)
    n_signers = len(set(ds.signers))
    print(f"{len(ds.y)} clips, {len(ds.labels)} signs, {n_signers} signer(s); chance = {1 / len(ds.labels):.0%}")
    for name, res in run(ds).items():
        si = res["signer_independent"]
        si_text = f"{si:.1%}" if si is not None else "n/a (needs 2+ signers)"
        print(f"\n{name}\n  signer-dependent   {res['signer_dependent']:.1%}\n  signer-independent {si_text}")
        for signer, acc in res["per_signer"].items():
            print(f"    held out {signer}: {acc:.1%}")


if __name__ == "__main__":
    main()
