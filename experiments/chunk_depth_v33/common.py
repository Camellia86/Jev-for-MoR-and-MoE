"""Protocol jev-chunk-depth-v3.3: paths, data roles, smoke limits, receipts, shards and stage markers.

Environment variables:
  JEV_HUGINN_DIR  model directory (read by experiments.shared_depth_v2.common)
  JEV_SCRATCH     node-local scratch for large intermediate files (tables, features)
  JEV_SMOKE=1     tiny end-to-end run; results go to separate *-smoke directories
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pyarrow.parquet as pq
import torch

from experiments.shared_depth_v2.common import ROOT, receipt, sha256_file, write_json
from experiments.chunk_depth_v31.data import DATA, OUT as OUT31, bucket

PROTOCOL = "jev-chunk-depth-v3.3"
SMOKE = os.environ.get("JEV_SMOKE") == "1"
_SUFFIX = "-smoke" if SMOKE else ""
CONFIG = ROOT / "docs" / "EXPERIMENT_CONFIG.v3.3.json"
RESULTS = ROOT / "results" / f"chunk-depth-v33{_SUFFIX}"
ART = ROOT / "artifacts" / f"chunk-depth-v33{_SUFFIX}"
SCRATCH = Path(os.environ.get("JEV_SCRATCH", ROOT / "artifacts" / f"chunk-depth-v33-scratch{_SUFFIX}"))
CODE_DIR = Path(__file__).resolve().parent

THETA_S = OUT31 / "pilot_SCHED-A_s17.pt"
HEAD32 = ROOT / "artifacts" / "chunk-depth-v32" / "head.pt"
R32 = ROOT / "artifacts" / "chunk-depth-v32" / "R.pt"
KV_JSON = ROOT / "results" / "chunk-depth-v31" / "kv.json"
REUSED = {"artifacts/chunk-depth-v31/pilot_SCHED-A_s17.pt": THETA_S, "artifacts/chunk-depth-v32/head.pt": HEAD32,
          "artifacts/chunk-depth-v32/R.pt": R32}

EXPECTED = {"fit": 3024, "calibration": 612, "confirm_train": 1087, "confirm_test": 1310, "btrain": 373}
SMOKE_LIMITS = {"fit": 10, "calibration": 6, "confirm_train": 3, "confirm_test": 3, "btrain": 16}
REPEATS = 300 if SMOKE else 10000
BOOT_SEED = 20261013
BUDGETS = (0.5, 0.6)
SEEDS = (17, 23, 31)
ARMS = ("G", "C", "D")
B_UPDATES = 2 if SMOKE else 93
FIXED_NU = 1.47


def load_config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _train_questions():
    return pq.read_table(DATA / "train.parquet").column("question").to_pylist()


def load_role(role: str) -> list:
    """fit, calibration, confirm (= confirm_train + confirm_test), confirm_train, confirm_test, btrain."""
    if role == "confirm":
        return load_role("confirm_train") + load_role("confirm_test")
    if role in ("fit", "confirm_train"):
        q = _train_questions()
        lo, hi = (10, 49) if role == "fit" else (50, 64)
        items = [it for it in torch.load(OUT31 / "items_formal_train.pt") if lo <= bucket(q[it["row"]]) <= hi]
    elif role == "calibration":
        items = torch.load(OUT31 / "items_pilot_cal.pt") + torch.load(OUT31 / "items_formal_cal.pt")
    elif role == "confirm_test":
        items = torch.load(OUT31 / "items_confirm.pt")
    elif role == "btrain":
        items = torch.load(OUT31 / "items_pilot_train.pt")
    else:
        raise KeyError(role)
    if SMOKE:
        return items[:SMOKE_LIMITS[role]]
    if len(items) != EXPECTED[role]:
        raise RuntimeError(f"role {role}: {len(items)} documents, expected {EXPECTED[role]}")
    return items


def confirm_part(index: int) -> str:
    n_train = SMOKE_LIMITS["confirm_train"] if SMOKE else EXPECTED["confirm_train"]
    return "train_50_64" if index < n_train else "test"


def shard_indices(n: int, shard: int, nshards: int) -> list[int]:
    return list(range(shard, n, nshards))


def save_atomic(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def code_hashes() -> dict:
    return {p.name: sha256_file(p) for p in sorted(CODE_DIR.glob("*.py"))}


def v33_receipt(kind: str, **fields) -> dict:
    return receipt(kind, protocol_v33=PROTOCOL, smoke=SMOKE, config_v33_sha256=sha256_file(CONFIG),
                   code_v33_sha256=code_hashes(), **fields)


def verify_reused(cfg: dict | None = None) -> dict:
    cfg = cfg or load_config()
    got = {rel: sha256_file(path) for rel, path in REUSED.items()}
    bad = {rel: h for rel, h in got.items() if h != cfg["reused_artifacts_sha256"][rel]}
    if bad:
        raise RuntimeError(f"reused artifact hash mismatch: {sorted(bad)}")
    return got


# ----------------------------------------------------------------------------- stage markers

def stage_path(name: str) -> Path:
    return RESULTS / "stages" / f"{name}.json"


def stage_done(name: str) -> bool:
    return stage_path(name).exists()


def mark_stage(name: str, **info) -> None:
    write_json(stage_path(name), {"stage": name, "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                  "node": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME"),
                                  "scratch": str(SCRATCH), **info})


def stage_info(name: str) -> dict:
    return json.loads(stage_path(name).read_text(encoding="utf-8"))


# ----------------------------------------------------------------------------- shard files

def shard_file(kind: str, shard: int, nshards: int, scratch: bool = False) -> Path:
    base = SCRATCH if scratch else ART / "shards"
    return base / f"{kind}_{shard}of{nshards}.pt"


def _done_marker(kind: str, shard: int, nshards: int, scratch: bool) -> Path:
    p = shard_file(kind, shard, nshards, scratch)
    return p.with_name(p.name + ".done")


def partial_file(kind: str, shard: int, nshards: int, scratch: bool = False) -> Path:
    p = shard_file(kind, shard, nshards, scratch)
    return p.with_name(p.name + ".partial")


def finish_shard(kind: str, shard: int, nshards: int, rows: list, scratch: bool = False) -> None:
    """rows: [(global_index, payload), ...]. Writes the shard, then its done marker, then drops the partial file."""
    save_atomic({"rows": rows}, shard_file(kind, shard, nshards, scratch))
    _done_marker(kind, shard, nshards, scratch).write_text(str(len(rows)), encoding="utf-8")
    partial_file(kind, shard, nshards, scratch).unlink(missing_ok=True)


def shard_is_done(kind: str, shard: int, nshards: int, scratch: bool = False) -> bool:
    return _done_marker(kind, shard, nshards, scratch).exists() and shard_file(kind, shard, nshards, scratch).exists()


def shards_complete(kind: str, nshards: int, scratch: bool = False) -> bool:
    return all(shard_is_done(kind, s, nshards, scratch) for s in range(nshards))


def load_shards(kind: str, nshards: int, scratch: bool = False) -> list:
    """Payloads of all shards ordered by global index; the indices must cover 0..n-1 exactly."""
    rows = []
    for s in range(nshards):
        if not shard_is_done(kind, s, nshards, scratch):
            raise RuntimeError(f"shard {kind} {s}/{nshards} is incomplete")
        rows.extend(torch.load(shard_file(kind, s, nshards, scratch), weights_only=False)["rows"])
    rows.sort(key=lambda r: r[0])
    idx = [r[0] for r in rows]
    if idx != list(range(len(idx))):
        raise RuntimeError(f"shards of {kind} do not cover 0..n-1")
    return [r[1] for r in rows]
