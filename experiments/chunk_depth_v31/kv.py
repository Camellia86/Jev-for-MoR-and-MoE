"""Pilot step 3 (plan v3.1 section 7.3): KV coverage correction on pilot_train, frozen model.

128 documents x 2 online rollouts with uniformly random line depths; token-weighted regression of
(realised - table) per-token NLL on shortfall s and excess e. Writes results/chunk-depth-v31/kv.json.
"""

from __future__ import annotations

import json

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v31.data import OUT, RESULTS
from experiments.chunk_depth_v31 import run as R


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    items = torch.load(OUT / "items_pilot_train.pt")[:128]
    feats = torch.load(OUT / "features_pilot_train.pt")[:128]
    model = H.load_model()
    g = torch.Generator().manual_seed(20261013)
    rows = []
    for item, f in zip(items, feats):
        _, lines, _ = R.lines_of(item)
        for _ in range(2):
            plan = [R.V3[int(i)] for i in torch.randint(0, 5, (len(lines),), generator=g)]
            nll, depths = R.retry(R.online, model, item, lambda z, dz, c, pl, h, plan=plan: plan[c - 1])
            hist = []
            for (a, b), d in zip(lines, depths):
                sc = R.scored_in(item, a, b)
                if sc and hist:
                    s, e = R.coverage(hist, d)
                    diff = float((nll[sc].double() - f["nll"][R.V3.index(d), sc].double()).mean())
                    rows.append({"s": s, "e": e, "delta_max": d - max(dj for _, dj in hist), "diff": diff, "n": len(sc)})
                hist.append((b - a, d))
    x = torch.tensor([[1.0, r["s"], r["e"]] for r in rows], dtype=torch.float64)
    w = torch.tensor([r["n"] for r in rows], dtype=torch.float64)
    y = torch.tensor([r["diff"] for r in rows], dtype=torch.float64)
    beta = torch.linalg.solve(x.T @ (w[:, None] * x), x.T @ (w * y))
    resid = y - x @ beta
    # sandwich standard errors with lines as units
    xtwx_inv = torch.linalg.inv(x.T @ (w[:, None] * x))
    meat = (x * (w * resid)[:, None]).T @ (x * (w * resid)[:, None])
    se = torch.sqrt(torch.diag(xtwx_inv @ meat @ xtwx_inv))
    out = {"lines": len(rows), "kappa_0": float(beta[0]), "kappa_s": float(beta[1]), "kappa_e": float(beta[2]),
           "se": {"kappa_0": float(se[0]), "kappa_s": float(se[1]), "kappa_e": float(se[2])},
           "residual_sd": float(resid.std())}
    groups = {"no_shortfall": [r for r in rows if r["s"] == 0], "shortfall_not_above_max":
              [r for r in rows if r["s"] > 0 and r["delta_max"] <= 0], "above_max": [r for r in rows if r["delta_max"] > 0]}
    out["groups"] = {k: {"lines": len(v), "token_weighted_mean": sum(r["diff"] * r["n"] for r in v) / max(1, sum(r["n"] for r in v))}
                     for k, v in groups.items()}
    write_json(RESULTS / "kv.json", receipt("chunk_depth_v3_1_pilot_kv_coverage", **out))
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
