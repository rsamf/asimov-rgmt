"""Shared test config.

NEBO_NO_STORE keeps pytest from writing nebo run files: the suite exercises
run_training / run_preprocess for real, which would otherwise mint a run per
test into .nebo/ (the 2026-07-27 cleanup archived ~70 pytest runs). nebo's
logging API still works — the file transport is just skipped.
"""
import os

os.environ.setdefault("NEBO_NO_STORE", "1")
