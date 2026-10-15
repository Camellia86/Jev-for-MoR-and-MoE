"""Phase 0 analysis: depth curves, question-level and chunk-level headroom, mixed-depth error."""

from __future__ import annotations

import json
import math

import torch

from experiments.shared_depth_v2 import cost as C
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v3.survey import DEPTHS, OUT, RESULTS

BUDGETS = (0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0)


def position_cost(t: int, d: int) -> float:
    """FLOPs to compute position t (0-based) at depth d with t+1 keys, plus its LM-head row."""
    return C.stack(1, t + 1, d) + C.lm_head(1)


def prepare():
    blob = torch.load(OUT / "survey_tables.pt")
    items, tables = blob["items"], blob["tables"]
    D = len(DEPTHS)
    units = []  # one entry per chunk: dict(item, chunk, nll[D, n_scored], cost[D], positions)
    for k, (item, tab) in enumerate(zip(items, tables)):
        S = len(item["ids"])
        chunk = item["chunk_of_pos"]
        first = item["first_solution_pos"]
        for c in range(item["n_chunks"]):
            pos = [t for t in range(S) if chunk[t] == c]
            scored = [t for t in pos if t <= S - 2 and t + 1 >= first]
            cost = torch.tensor([sum(position_cost(t, d) for t in pos) for d in DEPTHS], dtype=torch.float64)
            nll = tab[:, scored].double() if scored else torch.zeros(D, 0, dtype=torch.float64)
            units.append({"item": k, "chunk": c, "nll": nll, "cost": cost, "n_pos": len(pos),
                          "answer": c == item["answer_chunk"]})
    return items, tables, units


def lagrange(sel: torch.Tensor, ev: torch.Tensor, cost: torch.Tensor, budget_total: float) -> dict:
    """sel, ev: [U, D] summed NLL used for selection / evaluation; cost [U, D]. Randomised boundary."""
    def pick(nu):
        ch = (sel + nu * cost).argmin(1)
        idx = torch.arange(len(ch))
        return float(cost[idx, ch].sum()), float(ev[idx, ch].sum())

    c0, q0 = pick(0.0)
    if c0 <= budget_total:
        return {"cost": c0, "nll": q0}
    lo, hi = 0.0, 1e-12
    while pick(hi)[0] > budget_total:
        hi *= 2
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if pick(mid)[0] > budget_total:
            lo = mid
        else:
            hi = mid
    cl, ql = pick(lo)
    ch, qh = pick(hi)
    w = 0.0 if cl == ch else min(max((budget_total - ch) / (cl - ch), 0.0), 1.0)
    return {"cost": w * cl + (1 - w) * ch, "nll": w * ql + (1 - w) * qh}


def static_best(ev_by_depth: torch.Tensor, cost_by_depth: torch.Tensor, budget_total: float) -> dict:
    """ev_by_depth, cost_by_depth: [D] corpus totals. Best mixture of <= 2 depths within budget."""
    best = None
    D = len(ev_by_depth)
    for i in range(D):
        if cost_by_depth[i] <= budget_total:
            cand = (float(ev_by_depth[i]), float(cost_by_depth[i]), [(DEPTHS[i], 1.0)])
            best = cand if best is None or cand[0] < best[0] else best
        for j in range(D):
            if cost_by_depth[i] <= budget_total < cost_by_depth[j] and ev_by_depth[j] < ev_by_depth[i]:
                w = float((budget_total - cost_by_depth[i]) / (cost_by_depth[j] - cost_by_depth[i]))
                q = (1 - w) * float(ev_by_depth[i]) + w * float(ev_by_depth[j])
                cand = (q, budget_total, [(DEPTHS[i], 1 - w), (DEPTHS[j], w)])
                best = cand if best is None or cand[0] < best[0] else best
    if best is None:
        return None
    return {"nll": best[0], "cost": best[1], "support": best[2]}


def split_sum(nll: torch.Tensor, parity: int) -> torch.Tensor:
    return nll[:, parity::2].sum(1)


def headroom(units, items):
    D = len(DEPTHS)
    n_items = len(items)
    unit_cost = torch.stack([u["cost"] for u in units])
    base16 = float(unit_cost[:, DEPTHS.index(16)].sum())
    # question-level aggregates
    q_cost = torch.zeros(n_items, D, dtype=torch.float64)
    q_nll = torch.zeros(n_items, D, dtype=torch.float64)
    q_odd = torch.zeros(n_items, D, dtype=torch.float64)
    q_even = torch.zeros(n_items, D, dtype=torch.float64)
    for u in units:
        q_cost[u["item"]] += u["cost"]
        s = u["nll"].sum(1)
        q_nll[u["item"]] += s
        (q_odd if u["chunk"] % 2 else q_even)[u["item"]] += s
    c_full = torch.stack([u["nll"].sum(1) for u in units])
    c_odd = torch.stack([split_sum(u["nll"], 1) for u in units])
    c_even = torch.stack([split_sum(u["nll"], 0) for u in units])
    tokens = sum(u["nll"].shape[1] for u in units)
    tok_odd = sum(u["nll"][:, 1::2].shape[1] for u in units)
    tok_even = sum(u["nll"][:, 0::2].shape[1] for u in units)
    qtok_odd = sum(u["nll"].shape[1] for u in units if u["chunk"] % 2)
    qtok_even = tokens - qtok_odd
    rows = []
    for B in BUDGETS:
        total = B * base16
        st_full = static_best(q_nll.sum(0), q_cost.sum(0), total)
        if st_full is None:
            continue
        # static mixtures evaluated on the halves use the full-data static mixture weights
        def static_on(ev):
            return sum(w * float(ev[:, DEPTHS.index(d)].sum()) for d, w in st_full["support"])
        q_or = lagrange(q_nll, q_nll, q_cost, total)
        q_sh = (lagrange(q_odd, q_even, q_cost, total)["nll"] + lagrange(q_even, q_odd, q_cost, total)["nll"])
        c_or = lagrange(c_full, c_full, unit_cost, total)
        c_sh = (lagrange(c_odd, c_even, unit_cost, total)["nll"] + lagrange(c_even, c_odd, unit_cost, total)["nll"])
        static_tok = st_full["nll"] / tokens
        rows.append({
            "budget": B, "static_support": st_full["support"], "static_nll_per_token": static_tok,
            "question_oracle_gain": static_tok - q_or["nll"] / tokens,
            "question_split_gain": (static_on(q_even) + static_on(q_odd) - q_sh) / tokens,
            "chunk_oracle_gain": static_tok - c_or["nll"] / tokens,
            "chunk_split_gain": (static_on(c_even) + static_on(c_odd) - c_sh) / tokens,
        })
    curve = {d: float(q_nll[:, k].sum() / tokens) for k, d in enumerate(DEPTHS)}
    cost_curve = {d: float(q_cost[:, k].sum() / base16) for k, d in enumerate(DEPTHS)}
    ans = [u for u in units if u["answer"]]
    ans_tok = sum(u["nll"].shape[1] for u in ans)
    ans_curve = {d: float(sum(u["nll"][k].sum() for u in ans) / ans_tok) for k, d in enumerate(DEPTHS)}
    return {"tokens": tokens, "chunks": len(units), "items": n_items, "nll_curve": curve,
            "relative_cost": cost_curve, "answer_chunk_nll_curve": ans_curve, "rows": rows}


def per_chunk_spread(units):
    """How often each depth is the per-chunk argmin, and gain distributions between depths."""
    full = torch.stack([u["nll"].sum(1) / max(1, u["nll"].shape[1]) for u in units if u["nll"].shape[1] > 0])
    best = full.argmin(1)
    share = {d: float((best == k).double().mean()) for k, d in enumerate(DEPTHS)}
    gains = {}
    for a, b in ((4, 8), (8, 16), (16, 32), (32, 64)):
        g = full[:, DEPTHS.index(a)] - full[:, DEPTHS.index(b)]
        gains[f"{a}->{b}"] = {"mean": float(g.mean()), "sd": float(g.std()),
                              "q05": float(torch.quantile(g, 0.05)), "q95": float(torch.quantile(g, 0.95))}
    return {"argmin_share": share, "chunk_mean_gain": gains}


def mixed_error(items, tables):
    blob = torch.load(OUT / "survey_mixed.pt")
    rows_m, rows_u, deeper, shallower = [], [], [], []
    for m in blob["mixed"]:
        item, tab = items[m["index"]], tables[m["index"]]
        S = len(item["ids"])
        first = item["first_solution_pos"]
        scored = [t for t in range(S - 1) if t + 1 >= first]
        chunk = item["chunk_of_pos"]
        dm = torch.stack([m["mixed_nll"][t] - tab[DEPTHS.index(m["assign"][chunk[t]]), t] for t in scored])
        du = torch.stack([m["uniform16_nll"][t] - tab[DEPTHS.index(16), t] for t in scored])
        rows_m.append(float(dm.mean()))
        rows_u.append(float(du.mean()))
        for t, v in zip(scored, dm):
            c = chunk[t]
            prev_max = max(m["assign"][:c]) if c > 0 else m["assign"][0]
            (deeper if m["assign"][c] > prev_max else shallower).append(float(v))

    def stats(x):
        x = torch.tensor(x, dtype=torch.float64)
        return {"n": len(x), "mean": float(x.mean()), "sd": float(x.std()),
                "se": float(x.std() / math.sqrt(len(x)))}
    return {"items": len(rows_m), "mixed_minus_table_per_item": stats(rows_m),
            "uniform16_chunked_minus_table_per_item": stats(rows_u),
            "tokens_in_chunks_deeper_than_all_history": stats(deeper),
            "tokens_in_other_chunks": stats(shallower)}


def main() -> int:
    items, tables, units = prepare()
    out = {"headroom": headroom(units, items), "spread": per_chunk_spread(units),
           "mixed_depth_error": mixed_error(items, tables)}
    write_json(RESULTS / "survey_analysis.json", receipt("chunk_depth_v3_phase0_analysis", **out))
    h = out["headroom"]
    print("tokens", h["tokens"], "chunks", h["chunks"], "items", h["items"])
    print("nll curve", {d: round(v, 4) for d, v in h["nll_curve"].items()})
    print("answer nll", {d: round(v, 4) for d, v in h["answer_chunk_nll_curve"].items()})
    print("relative cost", {d: round(v, 3) for d, v in h["relative_cost"].items()})
    for r in h["rows"]:
        print(r["budget"], r["static_support"], {k: round(v, 4) for k, v in r.items()
                                                 if k.endswith("gain") or k == "static_nll_per_token"})
    print(json.dumps(out["spread"], indent=1))
    print(json.dumps(out["mixed_depth_error"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
