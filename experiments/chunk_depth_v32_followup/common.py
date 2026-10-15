from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.shared_depth_v2.common import ROOT, sha256_file, environment, write_json

OUT = ROOT / "artifacts" / "chunk-depth-v32-followup"
RESULTS = ROOT / "results" / "chunk-depth-v32-followup"
CONFIG_PATH = ROOT / "docs" / "EXPERIMENT_CONFIG.v3.2-followup.json"
CONFIG = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def verify_inputs():
    checked = {}
    for rel, wanted in CONFIG["input_sha256"].items():
        got = sha256_file(ROOT / rel)
        if got != wanted:
            raise RuntimeError(f"Input changed: {rel}: {got} != {wanted}")
        checked[rel] = got
    OUT.mkdir(parents=True, exist_ok=True)
    RESULTS.mkdir(parents=True, exist_ok=True)
    return checked


def check_time(start, cap_seconds):
    if time.perf_counter() - start > cap_seconds:
        raise RuntimeError("Predeclared runtime cap exceeded")


def record(kind, **fields):
    src = Path(__file__).parent
    return {"protocol_id": CONFIG["protocol"], "kind": kind,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "config_sha256": sha256_file(CONFIG_PATH),
            "source_sha256": {p.name: sha256_file(p) for p in sorted(src.glob("*.py"))},
            "input_sha256": CONFIG["input_sha256"], "environment": environment(), **fields}
