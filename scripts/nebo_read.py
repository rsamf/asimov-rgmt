"""Dump metric series from a .nebo run file (quick training-health checks).

Usage:
    uv run python scripts/nebo_read.py <run.nebo> [metric-substring ...]

With no substrings, lists all series names + counts. With substrings, prints
first/quintile/last points of every matching series.
"""
import sys

from nebo.core.fileformat import NeboFileReader
import collections

path = sys.argv[1]
pats = sys.argv[2:]

series = collections.defaultdict(list)
with open(path, "rb") as f:
    r = NeboFileReader(f)
    r.read_header()
    while True:
        e = r.read_next_entry()
        if e is None:
            break
        if e["type"] == "metric":
            p = e["payload"]
            series[p["name"]].append((p.get("step"), p["value"]))

if not pats:
    for k in sorted(series):
        print(f"{k:40s} n={len(series[k])}")
    sys.exit(0)

for k in sorted(series):
    if not any(pat in k for pat in pats):
        continue
    v = series[k]
    picks = [v[0]] + [v[max(0, len(v) * i // 5 - 1)] for i in range(1, 6)]
    line = "  ".join(
        f"@{s}:{val:.4g}" if isinstance(val, (int, float)) else f"@{s}:{val}"
        for s, val in picks
    )
    print(f"{k:32s} {line}")
