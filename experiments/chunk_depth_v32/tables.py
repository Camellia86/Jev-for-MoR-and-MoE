"""Stage 1 step 1 (plan v3.2 section 4.2): theta_S table runs.

For every document: whole-sequence teacher forcing at each candidate depth with the theta_S LoRA; per line
the mean NLL over scored tokens [lines, K] (float64) and the decision representation at p_{c-1} with the
whole prefix at that depth [K, lines, 5280] (bf16, the model's own precision, so storage is lossless).
Writes artifacts/chunk-depth-v32/tables_<role>.pt and results/chunk-depth-v32/tables.json.
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31.screen import Lora
from experiments.chunk_depth_v31.train import K, doc_structure
from experiments.chunk_depth_v32.common import OUT, RESULTS, THETA_S, check_theta_s, load_role


@torch.no_grad()
def table(model, item):
    ids = torch.tensor(item["ids"], device="cuda")[None]
    S = ids.shape[1]
    feats = H.coda_inputs(model, ids, R.item_state(item, S), R.V3)
    fc = H.freqs(model, torch.arange(S))
    st = doc_structure(item)
    pos = [l["a"] - 1 for l in st]
    nll, reps = [], []
    for d in R.V3:
        z = H.coda(model, feats[d], fc)[0]
        reps.append(z[pos].cpu())
        nll.append(torch.nn.functional.cross_entropy(H.head(model, z[None])[0, :-1], ids[0, 1:],
                                                     reduction="none").double().cpu())
    nll = torch.stack(nll)
    line_nll = torch.stack([nll[:, l["scored"]].mean(1) if l["n"] else torch.full((K,), float("nan"), dtype=torch.float64)
                            for l in st])
    return {"line_nll": line_nll, "reps": torch.stack(reps)}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roles", nargs="+", default=["readout_train", "calibration", "stage1_screen"])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    sha = check_theta_s()
    OUT.mkdir(parents=True, exist_ok=True)
    model = H.load_model()
    lora = Lora(model)
    lora.load(THETA_S)
    path = RESULTS / "tables.json"
    summary = json.loads(path.read_text(encoding="utf-8"))["roles"] if path.exists() else {}
    for role in args.roles:
        items = load_role(role)
        t0 = time.perf_counter()
        out = [R.retry(table, model, it) for it in items]
        torch.save(out, OUT / f"tables_{role}.pt")
        lines = torch.cat([o["line_nll"] for o in out])
        lines = lines[~lines.isnan().any(1)]
        summary[role] = {"documents": len(items), "lines_scored": len(lines), "seconds": time.perf_counter() - t0,
                         "mean_line_nll_by_depth": {d: float(lines[:, k].mean()) for k, d in enumerate(R.V3)}}
        print(json.dumps({role: summary[role]}), flush=True)
    write_json(path, receipt("chunk_depth_v3_2_tables", theta_s_sha256=sha, roles=summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
