"""Phase 0 probe addendum: does the boundary representation add information beyond position?

Same units, folds and calibration as probe.py; compares position-only, representation-only and
representation + position features (table evaluation), plus a nested alpha choice so that the
reported gain does not select alpha on the evaluation folds.
"""

from __future__ import annotations

import json

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v3.analyze_survey import static_best
from experiments.chunk_depth_v3.probe import (CHOICES, FOLDS, boundary_reps, crossfit, decision_units,
                                               position_features)
from experiments.chunk_depth_v3.survey import OUT, RESULTS

BUDGETS = (0.4, 0.5, 0.6)


def gain(choice, y, static_nll, n_tok):
    return (static_nll - float(y[torch.arange(len(y)), choice].sum())) / n_tok


def nested(features, units, y, cost, B, alphas, static_nll, n_tok, base16):
    """Choose alpha by inner cross-fitting on the training items of each outer fold."""
    n_items = max(u["item"] for u in units) + 1
    outer = (torch.arange(n_items) % FOLDS)[torch.tensor([u["item"] for u in units])]
    choice = torch.empty(len(units), dtype=torch.long)
    picked = []
    for f in range(FOLDS):
        tr = (outer != f).nonzero().flatten().tolist()
        te = (outer == f).nonzero().flatten().tolist()
        sub_units = [units[i] for i in tr]
        remap = {it: j for j, it in enumerate(sorted({u["item"] for u in sub_units}))}
        sub_units = [{**u, "item": remap[u["item"]]} for u in sub_units]
        best_alpha, best_gain = None, None
        for alpha in alphas:
            ch = crossfit(features[tr], sub_units, FOLDS, alpha, lambda c: B * float(c[:, CHOICES.index(16)].sum()))
            g = -float(y[tr][torch.arange(len(tr)), ch].sum())
            if best_gain is None or g > best_gain:
                best_alpha, best_gain = alpha, g
        picked.append(best_alpha)
        full_units = units
        ch_all = crossfit(features, full_units, FOLDS, best_alpha, lambda c: B * float(c[:, CHOICES.index(16)].sum()))
        choice[te] = ch_all[te]
    used = float(cost[torch.arange(len(y)), choice].sum()) / base16
    return {"gain": gain(choice, y, static_nll, n_tok), "relative_cost": used, "alphas": picked}


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    blob = torch.load(OUT / "survey_tables.pt")
    items, tables = blob["items"], blob["tables"]
    units = decision_units(items, tables)
    cached = OUT / "boundary_reps_depth8.pt"
    if cached.exists():
        reps = torch.load(cached).double()
    else:
        reps = boundary_reps(H.load_model(), items, units).double()
        torch.save(reps, cached)
    pos = position_features(units)
    y = torch.stack([u["nll"] for u in units])
    cost = torch.stack([u["cost"] for u in units])
    n_tok = sum(len(u["scored"]) for u in units)
    base16 = float(cost[:, CHOICES.index(16)].sum())
    feats = {"position_only": (pos, [0.1, 1.0, 10.0, 100.0]),
             "representation_only": (reps, [1000.0, 10000.0, 100000.0]),
             "representation_plus_position": (torch.cat([reps, pos * 30.0], 1), [1000.0, 10000.0, 100000.0])}
    rows = []
    for B in BUDGETS:
        st = static_best(y.sum(0), cost.sum(0), B * base16)
        row = {"budget": B, "static_support": st["support"]}
        for name, (f, alphas) in feats.items():
            row[name] = nested(f, units, y, cost, B, alphas, st["nll"], n_tok, base16)
        rows.append(row)
        print(json.dumps(row), flush=True)
    write_json(RESULTS / "survey_probe_combined.json",
               receipt("chunk_depth_v3_phase0_probe_combined", units=len(units), tokens=n_tok, rows=rows,
                       note="nested alpha selection; table evaluation; position features scaled by 30"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
