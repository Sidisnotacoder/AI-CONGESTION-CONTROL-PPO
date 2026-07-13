"""Versioned model saving.

train.py/train_real.py previously overwrote models/ppo_congestion.zip
and models/ppo_real_congestion.zip in place on every run, with no
record of which hyperparameters, timesteps, or code version produced
a given file. save_versioned() keeps that overwrite behavior (existing
loaders like evaluate_real.py still work unchanged) but additionally
writes a timestamped, never-overwritten copy plus a sidecar JSON of
the metadata needed to reproduce it.
"""

import json
import os
import subprocess
from datetime import datetime, timezone


def _git_commit_hash():
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
        ).stdout.strip() or None
    except Exception:
        return None


def save_versioned(model, base_path, metadata):
    """Save `model` to <base_path>.zip (fixed name, overwritten every
    run -- what existing loaders expect) and also to
    <base_path>_<timestamp>.zip (never overwritten) with a sidecar
    <base_path>_<timestamp>.json containing `metadata` plus the git
    commit hash and save timestamp.
    """
    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    versioned_base = f"{base_path}_{timestamp}"

    model.save(base_path)
    model.save(versioned_base)

    record = dict(metadata)
    record["timestamp_utc"] = timestamp
    record["git_commit"] = _git_commit_hash()

    with open(f"{versioned_base}.json", "w") as f:
        json.dump(record, f, indent=2)

    return versioned_base
