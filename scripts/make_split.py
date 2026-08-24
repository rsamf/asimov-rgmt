"""Build a train/test split JSON over a preprocessed cache's clips.

For a fresh corpus with no trained policy yet, stratify by dataset x
clip-length tertile (length is the best difficulty proxy available a priori,
since long clips fail more). Deterministic given --seed; the output matches
the schema train.py's eval.split_json consumes.

Usage:
    uv run python scripts/make_split.py \
        --cache cache/ \
        --out rgmt/data/splits/my_split.json \
        --test-frac 0.15 --seed 0
"""
import argparse
import json
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="preprocessed cache dir (reads manifest.json)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tertiles", type=int, default=3, help="length strata per dataset")
    a = ap.parse_args()

    import random
    rng = random.Random(a.seed)

    man = json.load(open(f"{a.cache}/manifest.json"))
    clips = man["clips"]  # each: {name, n_frames, ...}
    # dataset = first "__"-delimited token (ACCAD, CMU, ...)
    by_ds = defaultdict(list)
    for c in clips:
        by_ds[c["name"].split("__")[0]].append((c["name"], int(c["n_frames"])))

    train, test = [], []
    strata_meta = {}
    for ds, items in sorted(by_ds.items()):
        items.sort(key=lambda x: x[1])  # by length
        n = len(items)
        # split this dataset into length tertiles, sample test_frac from each
        # so both dataset AND length distributions are preserved in test.
        k = max(1, a.tertiles)
        picked_test = 0
        for t in range(k):
            lo = (t * n) // k
            hi = ((t + 1) * n) // k
            bucket = [nm for nm, _ in items[lo:hi]]
            rng.shuffle(bucket)
            n_test = round(len(bucket) * a.test_frac)
            test.extend(bucket[:n_test])
            train.extend(bucket[n_test:])
            picked_test += n_test
        strata_meta[ds] = {"clips": n, "test": picked_test}

    train.sort()
    test.sort()
    assert len(set(train) & set(test)) == 0
    assert len(train) + len(test) == len(clips)

    out = {
        "seed": a.seed,
        "test_frac": a.test_frac,
        "source": f"{a.cache} manifest.json ({len(clips)} clips)",
        "stratify": f"dataset x length-tertile ({a.tertiles})",
        "n_train": len(train),
        "n_test": len(test),
        "per_dataset": strata_meta,
        "train": train,
        "test": test,
    }
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"wrote {a.out}: {len(train)} train / {len(test)} test "
          f"({len(test)/len(clips)*100:.1f}% test) over {len(by_ds)} datasets")


if __name__ == "__main__":
    main()
