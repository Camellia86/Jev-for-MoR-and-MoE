"""Pilot step 2: frozen-core features (coda inputs at depths 4,6,8,12,16 over all positions) and frozen
per-position NLL tables. Layout: artifacts/chunk-depth-v31/features_<role>.pt = list of dicts
{"feats": bf16 [5, S, 5280], "nll": float32 [5, S-1]} aligned with items_<role>.pt."""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v31.data import OUT, RESULTS
from experiments.chunk_depth_v31.run import V3, item_state


@torch.no_grad()
def extract(model, item):
    ids = torch.tensor(item["ids"], device="cuda")[None]
    S = ids.shape[1]
    state = item_state(item, S)
    feats = H.coda_inputs(model, ids, state, V3)
    fc = H.freqs(model, torch.arange(S))
    nll = []
    for d in V3:
        logits = H.head(model, H.coda(model, feats[d], fc))[0, :-1]
        nll.append(torch.nn.functional.cross_entropy(logits, ids[0, 1:], reduction="none").float().cpu())
    return {"feats": torch.stack([feats[d][0] for d in V3]).cpu(), "nll": torch.stack(nll)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", nargs="+", default=["warm", "pilot_train", "pilot_cal", "pilot_screen"])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    model = H.load_model()
    summary = {}
    for role in args.roles:
        items = torch.load(OUT / f"items_{role}.pt")
        t0 = time.perf_counter()
        out = [extract(model, it) for it in items]
        torch.save(out, OUT / f"features_{role}.pt")
        tok = sum(o["nll"].shape[1] for o in out)
        summary[role] = {"documents": len(items), "seconds": time.perf_counter() - t0,
                         "frozen_token_nll_by_depth": {d: float(sum(o["nll"][k].sum() for o in out) / tok)
                                                       for k, d in enumerate(V3)}}
        print(json.dumps({role: summary[role]}), flush=True)
    path = RESULTS / "features.json"
    merged = json.loads(path.read_text(encoding="utf-8"))["roles"] if path.exists() else {}
    merged.update(summary)
    write_json(path, receipt("chunk_depth_v3_1_features", roles=merged))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
