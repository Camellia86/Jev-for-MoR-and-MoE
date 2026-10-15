"""Pilot stage 2 / formal stage 1 (protocol sections 5.6, 7.1): frozen-core feature cache.

For each document of the requested roles, stores BF16 coda inputs ln_f(raw state) for depths
4, 8, 16, 32 over the 255 teacher-forcing positions, plus the separate 128-position prefix-only
depth-4 coda input. Also stores the frozen base model's per-token target NLL at each depth
(stage warm_base_nll). Layout per role (artifacts, not committed):
  features/<role>.bin        uint16 view of bfloat16, shape [N, 4*255+128, 5280]
  features/<role>.json       cache header (protocol section 5.6)
  features/<role>.base_nll.pt float32 [N, 4, 128] per-token NLL of the frozen model
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time

import numpy as np
import torch

from . import huginn as H
from .common import (ARTIFACTS, MODEL_DIR, PROTOCOL_ID, RESULTS, Timer, git_identity, load_config,
                     receipt, sha256_file, write_json)
from .init_state import bf16_le_bytes

ROWS_PER_DOC = 4 * 255 + 128
HIDDEN = 5280


def feature_path(role: str):
    return ARTIFACTS / "features" / f"{role}.bin"


def open_features(role: str, n: int, mode: str = "r"):
    return np.memmap(feature_path(role), dtype=np.uint16, mode=mode, shape=(n, ROWS_PER_DOC, HIDDEN))


def load_features(role: str, index: int, header: dict | None = None) -> dict:
    """Return {depth: [255,5280] bf16, 'prefix': [128,5280] bf16} on CPU for one document."""
    header = header or json.loads((ARTIFACTS / "features" / f"{role}.json").read_text(encoding="utf-8"))
    mm = open_features(role, len(header["documents"]))
    block = torch.from_numpy(np.array(mm[index])).view(torch.bfloat16)
    out = {d: block[k * 255:(k + 1) * 255] for k, d in enumerate(H.DEPTHS)}
    out["prefix"] = block[4 * 255:]
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    cfg = load_config()
    win = torch.load(ARTIFACTS / "data" / "windows.pt")
    man = [json.loads(line) for line in open(ARTIFACTS / "data" / "manifest.jsonl", encoding="utf-8")]
    costs: dict = {}
    with Timer(costs, "model_load_seconds"):
        model = H.load_model()
    common = {
        "protocol_id": PROTOCOL_ID, "config_sha256": sha256_file(ARTIFACTS.parents[1] / "docs" / "EXPERIMENT_CONFIG.v2.json"),
        "model_revision": cfg["model"]["revision"], "tokenizer_sha256": sha256_file(MODEL_DIR / "tokenizer.json"),
        "dataset_revision": cfg["data"]["revision"], "parquet_sha256": cfg["data"]["sha256"],
        "initialization_replica": 0, "feature_kind": "coda_input_normalized_once",
        "layout": {"rows": ROWS_PER_DOC, "segments": [{"depth": d, "positions": 255} for d in H.DEPTHS]
                   + [{"depth": 4, "positions": 128, "prefix_only": True}]},
        "position_base": 0, "dtype": "bfloat16", "attention": "causal_full_no_padding",
        "stage": "frozen_core_feature", "code_sha256": sha256_file(ARTIFACTS.parents[1] / "experiments" / "shared_depth_v2" / "huginn.py"),
        "git": git_identity(),
    }
    summary = {}
    for role in args.roles:
        idx = [i for i, m in enumerate(man) if m["role"] == role]
        if args.limit:
            idx = idx[: args.limit]
        n = len(idx)
        (ARTIFACTS / "features").mkdir(parents=True, exist_ok=True)
        mm = open_features(role, n, "w+")
        base_nll = torch.empty(n, 4, 128, dtype=torch.float32)
        docs = []
        t_extract = t_write = t_state = 0.0
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for j, i in enumerate(idx):
            row = man[i]["source_row"]
            ids = win[i:i + 1, :255].long().cuda()
            targets = win[i, 128:256].long().cuda()
            t0 = time.perf_counter()
            state_cpu = H.initial_states([row], 255, 0, "cpu")
            t_state += time.perf_counter() - t0
            state = state_cpu.cuda()
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            feats = H.coda_inputs(model, ids, state)
            prefix = H.coda_inputs(model, ids[:, :128], state[:, :128], (4,))[4]
            fc = H.freqs(model, torch.arange(255))
            with torch.no_grad():
                for k, d in enumerate(H.DEPTHS):
                    logits = H.head(model, H.coda(model, feats[d], fc))[0, 127:]
                    base_nll[j, k] = torch.nn.functional.cross_entropy(logits, targets, reduction="none").cpu()
            block = torch.cat([feats[d][0] for d in H.DEPTHS] + [prefix[0]], 0).cpu()
            torch.cuda.synchronize()
            t_extract += time.perf_counter() - t0
            t0 = time.perf_counter()
            arr = block.view(torch.int16).numpy().view(np.uint16)
            mm[j] = arr
            docs.append({"source_row": row, "doc_id": man[i]["doc_id"],
                         "window_token_sha256": man[i]["window_token_sha256"],
                         "state_sha256": hashlib.sha256(bf16_le_bytes(state_cpu[0])).hexdigest(),
                         "payload_sha256": hashlib.sha256(arr.tobytes()).hexdigest()})
            t_write += time.perf_counter() - t0
            if (j + 1) % 128 == 0:
                print(f"{role}: {j + 1}/{n} docs, {time.perf_counter() - start:.0f}s", flush=True)
        mm.flush()
        del mm
        header = {**common, "role": role, "documents": docs, "manifest_indices": idx}
        write_json(ARTIFACTS / "features" / f"{role}.json", header)
        torch.save({"base_nll": base_nll, "depths": list(H.DEPTHS), "manifest_indices": idx,
                    "stage": "warm_base_nll"}, ARTIFACTS / "features" / f"{role}.base_nll.pt")
        elapsed = time.perf_counter() - start
        mean_nll = base_nll.mean(-1)
        summary[role] = {
            "documents": n, "seconds": elapsed, "seconds_per_doc": elapsed / n,
            "state_seconds": t_state, "gpu_seconds": t_extract, "write_hash_seconds": t_write,
            "bytes": feature_path(role).stat().st_size,
            "write_mib_per_second": feature_path(role).stat().st_size / 2**20 / max(t_write, 1e-9),
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
            "base_mean_nll_by_depth": {d: float(mean_nll[:, k].mean()) for k, d in enumerate(H.DEPTHS)},
            "header_sha256": sha256_file(ARTIFACTS / "features" / f"{role}.json"),
        }
        print(json.dumps({role: summary[role]}, indent=1), flush=True)
    previous = RESULTS / "feature_extraction.json"
    merged = json.loads(previous.read_text(encoding="utf-8"))["roles"] if previous.exists() else {}
    merged.update(summary)
    write_json(previous, receipt("feature_extraction", roles=merged, costs=costs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
