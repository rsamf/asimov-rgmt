"""Rebuild a difficulty-filtered train/test split for a new asimov-gmr release.

The 180 test names of an existing split (e.g. rgmt/data/splits/medium.json)
stay frozen; the train side is rebuilt from the release's compiled training
set and its difficulty labels, keeping only the requested difficulties. Use
after a re-retarget: difficulty labels move with the robot description.

Usage:
    uv run python scripts/make_difficulty_split.py \
        --release /mnt/.../runs/tune_v9 --base rgmt/data/splits/medium.json \
        --out rgmt/data/splits/medium_v9.json [--train-difficulties easy medium]
"""
import argparse
import json
from collections import Counter
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True,
                    help="asimov-gmr run dir with retargeted/clip_difficulty.json and train/*.npz")
    ap.add_argument("--base", required=True, help="split whose test names are kept frozen")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train-difficulties", nargs="+", default=["easy", "medium"])
    a = ap.parse_args()

    rel = Path(a.release)
    diff = json.load(open(rel / "retargeted" / "clip_difficulty.json"))
    labels = {n: v["difficulty"] for n, v in diff.items()}
    compiled = {p.stem for p in (rel / "train").glob("*.npz")}
    base = json.load(open(a.base))

    test = [n for n in base["test"] if n in compiled]
    missing = [n for n in base["test"] if n not in compiled]
    test_set = set(test)
    train = sorted(n for n in compiled
                   if n not in test_set and labels.get(n) in a.train_difficulties)

    per = {}
    for d in sorted(set(labels.values())):
        per[d] = {"train": sum(1 for n in train if labels[n] == d),
                  "test": sum(1 for n in test if labels.get(n) == d)}
    out = {
        "source": f"test names from {a.base}; train from {rel} ({', '.join(a.train_difficulties)})",
        "train_difficulties": list(a.train_difficulties),
        "n_train": len(train), "n_test": len(test),
        "per_difficulty": per,
        "difficulty": {n: labels[n] for n in train + test if n in labels},
        "train": train, "test": test,
    }
    Path(a.out).write_text(json.dumps(out, indent=1))
    print(f"wrote {a.out}: train {len(train)} test {len(test)}")
    print("per difficulty:", per)
    print("release label counts:", dict(Counter(labels.values())))
    if missing:
        print(f"WARNING: {len(missing)} frozen test clips are not in the release: {missing[:5]}")


if __name__ == "__main__":
    main()
