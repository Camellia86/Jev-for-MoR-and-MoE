"""Shared paths, configuration, hashing and receipt helpers for protocol jev-shared-depth-v2."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "docs" / "EXPERIMENT_CONFIG.v2.json"
RESULTS = ROOT / "results" / "shared-depth-v2"
ARTIFACTS = Path(os.environ.get("JEV_ARTIFACTS", ROOT / "artifacts" / "shared-depth-v2"))
MODEL_DIR = Path(os.environ.get("JEV_HUGINN_DIR", r"C:\Users\z5885\git\loopjev\checkpoints\huginn-0125"))
PARQUET = Path(os.environ.get(
    "JEV_PARQUET", r"C:\Users\z5885\git\loopjev\data\mor-fineweb-edu\train-00000-of-00234.parquet"))

PROTOCOL_ID = "jev-shared-depth-v2"


def load_config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk: int = 1 << 24) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def git_identity() -> dict:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()

    status = run("status", "--porcelain")
    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(status),
            "dirty_files": [line[3:] for line in status.splitlines()][:50]}


def environment() -> dict:
    info = {"python": sys.version.split()[0], "platform": platform.platform()}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda"] = torch.version.cuda
            info["device"] = torch.cuda.get_device_name(0)
            info["device_total_mib"] = torch.cuda.get_device_properties(0).total_memory // 2**20
    except ImportError:
        pass
    try:
        import transformers
        info["transformers"] = transformers.__version__
    except ImportError:
        pass
    return info


def receipt(kind: str, **fields) -> dict:
    return {"protocol_id": PROTOCOL_ID, "kind": kind,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(CONFIG_PATH), "git": git_identity(),
            "environment": environment(), **fields}


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    os.replace(tmp, path)


class Timer:
    """Accumulate wall-clock seconds into a dict entry."""

    def __init__(self, store: dict, key: str):
        self.store, self.key = store, key

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.store[self.key] = self.store.get(self.key, 0.0) + time.perf_counter() - self.start
        return False
