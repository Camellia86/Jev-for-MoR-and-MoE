"""Launch experiments/chunk_depth_v32_followup/evaluate.py unchanged, with retries on transient write errors.

The first run computed the full calibration and then failed with a transient PermissionError while creating
results/chunk-depth-v32-followup/calibration.json.tmp (the same path was writable a minute later). Online
execution is deterministic, so the rerun reproduces the calibration. This wrapper only retries JSON and
torch.save writes on PermissionError; it changes no computation and no file in the follow-up directory.
"""

from __future__ import annotations

import time

import torch

from experiments.chunk_depth_v32_followup import common as C
from experiments.chunk_depth_v32_followup import evaluate as E


def retrying(fn, name, attempts=20, wait=3.0):
    def wrapped(*args, **kwargs):
        for i in range(attempts):
            try:
                return fn(*args, **kwargs)
            except PermissionError as exc:
                print(f"{name}: transient PermissionError ({exc}); retry {i + 1}/{attempts}", flush=True)
                time.sleep(wait)
        return fn(*args, **kwargs)
    return wrapped


if __name__ == "__main__":
    C.write_json = retrying(C.write_json, "write_json")
    torch.save = retrying(torch.save, "torch.save")
    E.main()
