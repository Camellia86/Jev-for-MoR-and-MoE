"""Phase 0 follow-up: headroom without the GSM8K '#### N' answer line, and realised mixed-depth gain.

Selection uses odd-indexed scored tokens of each chunk from the fixed-depth tables; evaluation uses
even-indexed tokens. The realised check executes the selected per-chunk depths with Huginn's
latest-m4 KV lookup on held-out items and compares against the static mixture.
"""

from __future__ import annotations

import argparse
import json

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v3.analyze_survey import lagrange, position_cost, static_best
from experiments.chunk_depth_v3.survey import DEPTHS, OUT, RESULTS, mixed_run

CHOICES = (4, 6, 8, 12, 16, 24, 32)
BUDGETS = (0.5, 0.75, 1.0)


def units_without_answer(items, tables):
    idx = [DEPTHS.index(d) for d in CHOICES]
    units = []
    for k, (item, tab) in enumerate(zip(items, tables)):
        S, chunk, first = len(item["ids"]), item["chunk_of_pos"], item["first_solution_pos"]
        for c in range(item["n_chunks"]):
            pos = [t for t in range(S) if chunk[t] == c]
            scored = [t for t in pos if t <= S - 2 and t + 1 >= first and c != item["answer_chunk"]
                      and chunk[t + 1] != item["answer_chunk"]]
            cost = torch.tensor([sum(position_cost(t, d) for t in pos) for d in CHOICES], dtype=torch.float64)
            nll = tab[idx][:, scored].double() if scored else torch.zeros(len(CHOICES), 0, dtype=torch.float64)
            units.append({"item": k, "chunk": c, "scored": scored, "nll": nll, "cost": cost})
    return units


def nu_for(sel, cost, total):
    lo, hi = 0.0, 1e-12

    def c_of(nu):
        ch = (sel + nu * cost).argmin(1)
        return float(cost[torch.arange(len(ch)), ch].sum())

    if c_of(0.0) <= total:
        return 0.0
    while c_of(hi) > total:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if c_of(mid) > total else (lo, mid)
    return hi


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--realised-items", type=int, default=96)
    args = parser.parse_args()
    blob = torch.load(OUT / "survey_tables.pt")
    items, tables = blob["items"], blob["tables"]
    units = units_without_answer(items, tables)
    cost = torch.stack([u["cost"] for u in units])
    full = torch.stack([u["nll"].sum(1) for u in units])
    odd = torch.stack([u["nll"][:, 1::2].sum(1) for u in units])
    even = torch.stack([u["nll"][:, 0::2].sum(1) for u in units])
    n_tok = sum(u["nll"].shape[1] for u in units)
    n_even = sum(u["nll"][:, 0::2].shape[1] for u in units)
    base16 = float(cost[:, CHOICES.index(16)].sum())
    item_of = torch.tensor([u["item"] for u in units])
    curve = {d: float(full[:, k].sum() / n_tok) for k, d in enumerate(CHOICES)}

    # question-level aggregates for the same token set
    n_items = len(items)
    q_full = torch.zeros(n_items, len(CHOICES), dtype=torch.float64).index_add_(0, item_of, full)
    q_cost = torch.zeros(n_items, len(CHOICES), dtype=torch.float64).index_add_(0, item_of, cost)
    q_odd_chunks = torch.zeros_like(q_full)
    q_even_chunks = torch.zeros_like(q_full)
    for u, s in zip(units, full):
        (q_odd_chunks if u["chunk"] % 2 else q_even_chunks)[u["item"]] += s

    table_rows = []
    for B in BUDGETS:
        total = B * base16
        st = static_best(full.sum(0), cost.sum(0), total)
        st_even = sum(w * float(even[:, CHOICES.index(d)].sum()) for d, w in st["support"])
        st_q = lambda ev: sum(w * float(ev[:, CHOICES.index(d)].sum()) for d, w in st["support"])
        c_split = lagrange(odd, even, cost, total)
        q_split = (lagrange(q_odd_chunks, q_even_chunks, q_cost, total)["nll"]
                   + lagrange(q_even_chunks, q_odd_chunks, q_cost, total)["nll"])
        table_rows.append({
            "budget": B, "static_support": st["support"], "static_nll_per_token": st["nll"] / n_tok,
            "chunk_oracle_gain": (st["nll"] - lagrange(full, full, cost, total)["nll"]) / n_tok,
            "chunk_split_gain_even_tokens": (st_even - c_split["nll"]) / n_even,
            "question_split_gain": (st_q(q_even_chunks) + st_q(q_odd_chunks) - q_split) / n_tok,
        })

    torch.backends.cuda.matmul.allow_tf32 = False
    model = H.load_model()
    held = list(range(len(items) - args.realised_items, len(items)))
    realised = []
    for B in BUDGETS:
        total = B * base16
        nu = nu_for(odd, cost, total)
        st = static_best(full.sum(0), cost.sum(0), total)
        choice = (odd + nu * cost).argmin(1)
        gains, tok, cost_dyn, cost_static = [], 0, 0.0, 0.0
        for k in held:
            sel = [i for i in range(len(units)) if units[i]["item"] == k]
            depth_of_chunk = [16] * items[k]["n_chunks"]
            for i in sel:
                depth_of_chunk[units[i]["chunk"]] = CHOICES[int(choice[i])]
            mixed = mixed_run(model, items[k], depth_of_chunk, "cuda")
            for i in sel:
                ev_pos = units[i]["scored"][0::2]
                if not ev_pos:
                    continue
                dyn = float(mixed[ev_pos].double().sum())
                stat = sum(w * float(tables[k][DEPTHS.index(d), ev_pos].double().sum()) for d, w in st["support"])
                gains.append((stat - dyn, len(ev_pos)))
                tok += len(ev_pos)
                cost_dyn += float(cost[i, choice[i]])
                cost_static += sum(w * float(cost[i, CHOICES.index(d)]) for d, w in st["support"])
        per_item = torch.tensor([g for g, _ in gains], dtype=torch.float64)
        realised.append({"budget": B, "nu": nu, "held_out_items": len(held), "tokens": tok,
                         "realised_gain_per_token": float(per_item.sum() / tok),
                         "dynamic_relative_cost": cost_dyn / base16 * len(units) / sum(
                             1 for u in units if u["item"] in held),
                         "cost_ratio_dynamic_over_static": cost_dyn / cost_static,
                         "depth_share": {d: float((choice == j).double().mean()) for j, d in enumerate(CHOICES)}})
        print(json.dumps(realised[-1]), flush=True)
    out = {"choices": CHOICES, "tokens": n_tok, "nll_curve_without_answer_line": curve,
           "table_estimates": table_rows, "realised_latest_m4": realised}
    write_json(RESULTS / "survey_realised.json", receipt("chunk_depth_v3_phase0_realised", **out))
    print(json.dumps({"curve": curve, "table": table_rows}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
