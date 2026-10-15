"""Protocol jev-chunk-depth-v3.2 (docs/EXPERIMENT_PLAN.v3.2.zh-CN.md): paths, data roles and theta_S."""

from __future__ import annotations

import pyarrow.parquet as pq
import torch

from experiments.shared_depth_v2.common import ROOT, sha256_file
from experiments.chunk_depth_v31.data import DATA, OUT as OUT31, bucket

OUT = ROOT / "artifacts" / "chunk-depth-v32"
RESULTS = ROOT / "results" / "chunk-depth-v32"
THETA_S = OUT31 / "pilot_SCHED-A_s17.pt"
THETA_S_SHA256 = "833160b2f0bd5c3001106407ceb2e87434683032efe11408c44bb1b47ce63625"
SCREEN_BUCKETS = {"stage1_screen": (90, 94), "stage2_screen": (95, 99)}


def check_theta_s():
    got = sha256_file(THETA_S)
    if got != THETA_S_SHA256:
        raise RuntimeError(f"theta_S checkpoint hash mismatch: {got}")
    return got


def load_role(role: str):
    if role == "readout_train":
        return torch.load(OUT31 / "items_formal_train.pt")
    if role == "calibration":
        return torch.load(OUT31 / "items_pilot_cal.pt") + torch.load(OUT31 / "items_formal_cal.pt")
    if role == "development":
        return torch.load(OUT31 / "items_pilot_screen.pt")
    if role in SCREEN_BUCKETS:
        questions = pq.read_table(DATA / "train.parquet").column("question").to_pylist()
        lo, hi = SCREEN_BUCKETS[role]
        return [it for it in torch.load(OUT31 / "items_formal_screen.pt") if lo <= bucket(questions[it["row"]]) <= hi]
    raise KeyError(role)
