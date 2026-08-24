"""Wait for a NEW nebo alert — a high-water-mark wrapper around `nebo alerts ls`.

`nebo runs wait` (0.3.0) wakes on pre-existing alerts, so a supervisor that
re-arms after handling an alert wakes instantly on the one it just handled.
This polls the daemon's alert list and exits only when an alert STRICTLY NEWER
than --since arrives (for the given run, or any run if omitted).

Prints the newest matching alert as JSON and exits 0; on timeout prints
{"status": "timeout"} and exits 0 (matching `nebo runs wait` semantics).

Usage:
    uv run python scripts/nebo_wait_new.py --since 1784181444.55 \
        [--run RUN_ID] [--min-level 20] [--timeout 1800] [--poll 60]
"""
import argparse
import json
import subprocess
import sys
import time


def fetch_alerts() -> list[dict]:
    """Flatten `alerts ls` into one fired-event stream.

    The listing mixes two shapes: code-fired alerts (nb.alert — have a
    `timestamp`) and RULES (have a `condition` and carry their own firing
    history in a `fired` array). Normalize both so the caller sees a single
    list of {title, level, timestamp, run_id, ...} events.
    """
    out = subprocess.run(
        ["uv", "run", "nebo", "alerts", "ls", "--json"],
        capture_output=True, text=True, timeout=120,
    )
    if out.returncode != 0:
        return []
    try:
        entries = json.loads(out.stdout).get("alerts", [])
    except json.JSONDecodeError:
        return []
    events = []
    for a in entries:
        if "timestamp" in a:            # code-fired alert
            events.append(a)
        for f in a.get("fired") or []:  # rule firing history
            ev = dict(f)
            ev.setdefault("title", a.get("title"))
            ev.setdefault("level", a.get("level", 20))
            ev.setdefault("rule_id", a.get("id"))
            events.append(ev)
    return events


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", type=float, required=True,
                    help="only alerts with timestamp strictly greater match")
    ap.add_argument("--run", default=None, help="restrict to this run id")
    ap.add_argument("--min-level", type=int, default=20)
    ap.add_argument("--timeout", type=float, default=1800, help="seconds")
    ap.add_argument("--poll", type=float, default=60, help="poll interval s")
    args = ap.parse_args()

    deadline = time.monotonic() + args.timeout
    while True:
        fresh = [
            a for a in fetch_alerts()
            if a.get("timestamp", 0) > args.since
            and a.get("level", 0) >= args.min_level
            and (args.run is None or a.get("run_id") in (None, args.run))
        ]
        if fresh:
            fresh.sort(key=lambda a: a["timestamp"])
            print(json.dumps({"status": "alert", "alerts": fresh}))
            return
        if time.monotonic() >= deadline:
            print(json.dumps({"status": "timeout"}))
            return
        time.sleep(min(args.poll, max(0.0, deadline - time.monotonic())))


if __name__ == "__main__":
    main()
