"""Phase 0 predictability probe: can the context before a chunk predict its depth profile?

For every solution chunk c (excluding the question chunk and the '#### N' answer line), the
decision representation is ln_f(coda(.)) of the frozen depth-8 run at the last position of chunk
c-1. A ridge regression (5 item-level folds) predicts the chunk's summed NLL at each candidate
depth; the policy picks argmin(pred + nu * cost) with nu calibrated on the training folds to the
budget. Evaluation: fixed-depth tables (all scored tokens of the chunk) and a realised
latest-m4 mixed-depth run. A position-only predictor (chunk index, previous chunk length) is
the baseline.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v3.analyze_survey import position_cost, static_best
from experiments.chunk_depth_v3.realized import CHOICES, nu_for
from experiments.chunk_depth_v3.survey import DEPTHS, OUT, RESULTS, mixed_run, state_matrix

REF_DEPTH = 8
FOLDS = 5


def decision_units(items, tables):
    idx = [DEPTHS.index(d) for d in CHOICES]
    units = []
    for k, (item, tab) in enumerate(zip(items, tables)):
        S, chunk = len(item["ids"]), item["chunk_of_pos"]
        for c in range(1, item["answer_chunk"]):
            pos = [t for t in range(S) if chunk[t] == c]
            scored = [t for t in pos if t <= S - 2 and chunk[t + 1] != item["answer_chunk"]]
            if not scored:
                continue
            prev = [t for t in range(S) if chunk[t] == c - 1]
            cost = torch.tensor([sum(position_cost(t, d) for t in pos) for d in CHOICES], dtype=torch.float64)
            units.append({"item": k, "chunk": c, "boundary": prev[-1], "prev_len": len(prev),
                          "scored": scored, "nll": tab[idx][:, scored].double().sum(1), "cost": cost})
    return units


@torch.no_grad()
def boundary_reps(model, items, units):
    by_item = {}
    for i, u in enumerate(units):
        by_item.setdefault(u["item"], []).append(i)
    reps = torch.empty(len(units), 5280, dtype=torch.float32)
    for k, idxs in by_item.items():
        ids = torch.tensor(items[k]["ids"], device="cuda")[None]
        S = ids.shape[1]
        fc = H.freqs(model, torch.arange(S))
        state = state_matrix(items[k]["row"], S).cuda()[None]
        feats = H.coda_inputs(model, ids, state, (REF_DEPTH,))
        z = H.coda(model, feats[REF_DEPTH], fc)[0].float()
        for i in idxs:
            reps[i] = z[units[i]["boundary"]].cpu()
    return reps


def ridge_predict(x_tr, y_tr, x_te, alpha):
    mu, sd = x_tr.mean(0), x_tr.std(0) + 1e-6
    a, b = (x_tr - mu) / sd, (x_te - mu) / sd
    ym = y_tr.mean(0)
    k = a @ a.T
    w = torch.linalg.solve(k + alpha * torch.eye(len(a), dtype=a.dtype), y_tr - ym)
    return b @ (a.T @ w) + ym


def position_features(units):
    f = []
    for u in units:
        onehot = [1.0 if u["chunk"] == j else 0.0 for j in range(1, 9)] + [1.0 if u["chunk"] >= 9 else 0.0]
        f.append(onehot + [u["prev_len"] / 50.0])
    return torch.tensor(f, dtype=torch.float64)


def crossfit(features, units, folds, alpha, budget_total_fn):
    """Return per-unit chosen depth index; nu calibrated on training folds."""
    n_items = max(u["item"] for u in units) + 1
    fold_of_item = torch.arange(n_items) % folds
    fold = fold_of_item[torch.tensor([u["item"] for u in units])]
    y = torch.stack([u["nll"] for u in units])
    cost = torch.stack([u["cost"] for u in units])
    choice = torch.empty(len(units), dtype=torch.long)
    for f in range(folds):
        tr, te = fold != f, fold == f
        pred_tr = ridge_predict(features[tr], y[tr], features[tr], alpha)
        pred_te = ridge_predict(features[tr], y[tr], features[te], alpha)
        nu = nu_for(pred_tr, cost[tr], budget_total_fn(cost[tr]))
        choice[te] = (pred_te + nu * cost[te]).argmin(1)
    return choice


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.4, 0.5, 0.6])
    parser.add_argument("--alpha", type=float, nargs="+", default=[100.0, 1000.0, 10000.0])
    parser.add_argument("--realised-items", type=int, default=720)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    blob = torch.load(OUT / "survey_tables.pt")
    items, tables = blob["items"], blob["tables"]
    units = decision_units(items, tables)
    model = H.load_model()
    reps = boundary_reps(model, items, units).double()
    pos_f = position_features(units)
    y = torch.stack([u["nll"] for u in units])
    cost = torch.stack([u["cost"] for u in units])
    n_tok = sum(len(u["scored"]) for u in units)
    base16 = float(cost[:, CHOICES.index(16)].sum())
    rows = []
    for B in args.budgets:
        st = static_best(y.sum(0), cost.sum(0), B * base16)
        static_nll = st["nll"]
        oracle_choice = (y + nu_for(y, cost, B * base16) * cost).argmin(1)
        entry = {"budget": B, "static_support": st["support"], "static_nll_per_token": static_nll / n_tok,
                 "in_sample_oracle_gain": (static_nll - float(y[torch.arange(len(y)), oracle_choice].sum())) / n_tok}
        for name, feats, alphas in (("representation", reps, args.alpha), ("position_only", pos_f, [1.0, 10.0])):
            best = None
            for alpha in alphas:
                ch = crossfit(feats, units, FOLDS, alpha, lambda c: B * float(c[:, CHOICES.index(16)].sum()))
                q = float(y[torch.arange(len(y)), ch].sum())
                c_used = float(cost[torch.arange(len(y)), ch].sum()) / base16
                res = {"alpha": alpha, "table_gain": (static_nll - q) / n_tok, "relative_cost": c_used,
                       "depth_share": {d: float((ch == j).double().mean()) for j, d in enumerate(CHOICES)},
                       "choice": ch}
                if best is None or res["table_gain"] > best["table_gain"]:
                    best = res
            entry[name] = best
        rows.append(entry)
        print(json.dumps({k: (v if not isinstance(v, dict) else {kk: vv for kk, vv in v.items() if kk != "choice"})
                          for k, v in entry.items()}), flush=True)

    # realised latest-m4 check for the representation probe at the middle budget (alpha chosen above;
    # alpha selection used held-out folds, so the realised figure is mildly optimistic)
    realised = []
    for entry in rows:
        ch = entry["representation"]["choice"]
        st = entry["static_support"]
        gain, tok, cdyn = 0.0, 0, 0.0
        by_item = {}
        for i, u in enumerate(units):
            by_item.setdefault(u["item"], []).append(i)
        for k in list(by_item)[: args.realised_items]:
            depth_of_chunk = [16] * items[k]["n_chunks"]
            for i in by_item[k]:
                depth_of_chunk[units[i]["chunk"]] = CHOICES[int(ch[i])]
            mixed = mixed_run(model, items[k], depth_of_chunk, "cuda")
            for i in by_item[k]:
                sc = units[i]["scored"]
                dyn = float(mixed[sc].double().sum())
                stat = sum(w * float(tables[k][DEPTHS.index(d), sc].double().sum()) for d, w in st)
                gain += stat - dyn
                tok += len(sc)
                cdyn += float(cost[i, ch[i]])
        realised.append({"budget": entry["budget"], "items": min(args.realised_items, len(by_item)), "tokens": tok,
                         "realised_gain_per_token": gain / tok})
        print(json.dumps(realised[-1]), flush=True)
    for entry in rows:
        for name in ("representation", "position_only"):
            entry[name].pop("choice")
    write_json(RESULTS / "survey_probe.json", receipt("chunk_depth_v3_phase0_probe", ref_depth=REF_DEPTH,
               folds=FOLDS, choices=CHOICES, units=len(units), tokens=n_tok, rows=rows, realised_latest_m4=realised))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
