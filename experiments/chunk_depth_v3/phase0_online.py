"""Phase 0 redo for protocol v3.1: online, pre-decision-only policies under the v3 execution semantics.

Setting (matches the formal protocol): question chunk at depth 16 and outside the budget; decision units
are solution lines (the GSM8K '#### N' line is never scored); candidate depths {4, 6, 8, 12, 16};
mixed depth executed with Huginn's latest-m4 KV lookup; every policy and comparator is run for real.

Items (phase-0 survey, 720 GSM8K train problems) are split by index: fit 0-299, cal 300-419, test 420-719.

Policies
  static     best mixture of <= 2 uniform line depths, chosen on cal from real runs
  position   ridge on position features (line index one-hot, previous line length) -> per-token NLL per
             depth; a position schedule family indexed by nu, nu chosen on cal by real execution
  rep        ridge on [coda output at the decision position, depth of that output, position features]
             -> per-token NLL per depth; decided online from the actual mixed-history representation
  question   original idea: one depth per problem from the representation at the end of the question
Decision rule: argmin_d  predicted_per_token_nll(d) + nu * r(d), r = relative per-token cost (no line length).
Calibration: scan nu on cal with real execution; keep the lowest-NLL nu whose realised cost <= budget.
KV study: random per-line depths on fit items; realised-minus-table per-token NLL vs history coverage.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import MODEL_DIR, receipt, write_json
from experiments.chunk_depth_v3.analyze_survey import position_cost
from experiments.chunk_depth_v3.survey import DEPTHS, OUT, RESULTS, state_matrix

V3 = (4, 6, 8, 12, 16)
R_REL = torch.tensor([0.301, 0.418, 0.534, 0.767, 1.000], dtype=torch.float64)
NU_GRID = (0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0, 6.0)
FIT, CAL, TEST = range(0, 300), range(300, 420), range(420, 720)


# ----------------------------------------------------------------------------- item structure

def lines_of(item):
    """Return question end (exclusive), list of (start, end) position ranges for decision lines, and S."""
    S, chunk = len(item["ids"]), item["chunk_of_pos"]
    lines = []
    for c in range(1, item["answer_chunk"]):
        pos = [t for t in range(S) if chunk[t] == c]
        if pos:
            lines.append((pos[0], pos[-1] + 1))
    return lines[0][0], lines, S


def scored_in(item, a, b):
    S, chunk, ans = len(item["ids"]), item["chunk_of_pos"], item["answer_chunk"]
    return [t for t in range(a, b) if t <= S - 2 and chunk[t + 1] != ans]


def pos_feats(c, prev_len):
    return [1.0 if min(c, 9) == i else 0.0 for i in range(1, 10)] + [prev_len / 50.0]


def line_cost(a, b, d):
    return sum(position_cost(t, d) for t in range(a, b))


# ----------------------------------------------------------------------------- execution

def new_cache():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("raven_modeling_minimal.HuginnDynamicCache", str(MODEL_DIR))
    return cls(lookup_strategy="latest-m4")


@torch.no_grad()
def run_segment(model, cache, ids, state, a, b, d):
    fc = H.freqs(model, torch.arange(a, b))
    e = H.prelude(model, ids[:, a:b], fc, cache)
    r = H.core_steps(model, state[:, a:b], e, fc, 0, d, cache)
    z = H.coda(model, H.ln_f(model, r), fc, cache)
    return z[0], H.head(model, z)[0]


@torch.no_grad()
def online(model, item, policy):
    """Run question at 16, then decide each line from the actual representation. Returns per-position NLL,
    chosen depths and the representation used for each decision."""
    q_end, lines, S = lines_of(item)
    ids = torch.tensor(item["ids"], device="cuda")[None]
    state = state_matrix(item["row"], S).cuda()[None]
    cache = new_cache()
    nll = torch.full((S - 1,), float("nan"))

    def record(a, b, logits):
        hi = min(b, S - 1)
        if hi > a:
            nll[a:hi] = torch.nn.functional.cross_entropy(logits[: hi - a].float(), ids[0, a + 1:hi + 1],
                                                          reduction="none").cpu()

    z, logits = run_segment(model, cache, ids, state, 0, q_end, 16)
    record(0, q_end, logits)
    last_z, last_d, prev_len = z[-1].float().cpu(), 16, q_end
    depths = []
    for c, (a, b) in enumerate(lines, start=1):
        d = policy(last_z, last_d, c, prev_len)
        z, logits = run_segment(model, cache, ids, state, a, b, d)
        record(a, b, logits)
        depths.append(d)
        last_z, last_d, prev_len = z[-1].float().cpu(), d, b - a
    return nll, depths


def doc_stats(item, nll, depths):
    _, lines, _ = lines_of(item)
    sc = [t for (a, b) in lines for t in scored_in(item, a, b)]
    vals = nll[sc].double()
    cost = sum(line_cost(a, b, d) for (a, b), d in zip(lines, depths))
    base = sum(line_cost(a, b, 16) for (a, b) in lines)
    return {"mean_nll": float(vals.mean()), "sum_nll": float(vals.sum()), "tokens": len(sc),
            "cost": cost, "base16": base, "depths": depths}


def retry(fn, *args):
    for attempt in range(6):
        try:
            return fn(*args)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("repeated OOM")


# ----------------------------------------------------------------------------- predictors

class Ridge:
    def __init__(self, x, y, alpha):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        a = torch.cat([(x - self.mu) / self.sd, torch.ones(len(x), 1, dtype=x.dtype)], 1)
        eye = torch.eye(a.shape[1], dtype=x.dtype)
        eye[-1, -1] = 0
        self.w = torch.linalg.solve(a.T @ a + alpha * eye, a.T @ y)

    def __call__(self, x):
        a = torch.cat([(x - self.mu) / self.sd, torch.ones(len(x), 1, dtype=x.dtype)], 1)
        return a @ self.w


def fit_ridge(x, y, groups, alphas):
    """Choose alpha by 5-fold grouped CV on prediction MSE, then refit on all rows."""
    best = None
    for alpha in alphas:
        err = 0.0
        for f in range(5):
            tr, te = groups % 5 != f, groups % 5 == f
            err += float(((Ridge(x[tr], y[tr], alpha)(x[te]) - y[te]) ** 2).sum())
        if best is None or err < best[1]:
            best = (alpha, err)
    return Ridge(x, y, best[0]), best[0]


@torch.no_grad()
def fit_rows(model, items, tables, idx):
    """Training rows: for each line and each previous-line depth d', the representation at the decision
    position from the fixed-depth-d' run (question-end decisions use d'=16 only), targets = per-token
    table NLL of the line at each candidate depth."""
    rows_x, rows_p, rows_y, groups, q_x, q_y, q_g = [], [], [], [], [], [], []
    tab_idx = [DEPTHS.index(d) for d in V3]
    for k in idx:
        item = items[k]
        q_end, lines, S = lines_of(item)
        ids = torch.tensor(item["ids"], device="cuda")[None]
        state = state_matrix(item["row"], S).cuda()[None]
        fc = H.freqs(model, torch.arange(S))
        feats = H.coda_inputs(model, ids, state, V3)
        zs = {d: H.coda(model, feats[d], fc)[0].float().cpu() for d in V3}
        tab = tables[k][tab_idx].double()
        all_sc = []
        prev_len = q_end
        for c, (a, b) in enumerate(lines, start=1):
            sc = scored_in(item, a, b)
            if sc:
                y = tab[:, sc].mean(1)
                for dprev in (V3 if c > 1 else (16,)):
                    onehot = [1.0 if dprev == d else 0.0 for d in V3]
                    rows_x.append(torch.cat([zs[dprev][a - 1].double(), torch.tensor(onehot, dtype=torch.float64)]))
                    rows_p.append(pos_feats(c, prev_len))
                    rows_y.append(y)
                    groups.append(k)
            all_sc += sc
            prev_len = b - a
        if all_sc:
            q_x.append(zs[16][q_end - 1].double())
            q_y.append(tab[:, all_sc].mean(1))
            q_g.append(k)
    return (torch.stack(rows_x), torch.tensor(rows_p, dtype=torch.float64), torch.stack(rows_y),
            torch.tensor(groups), torch.stack(q_x), torch.stack(q_y), torch.tensor(q_g))


# ----------------------------------------------------------------------------- evaluation helpers

def aggregate(stats):
    n = len(stats)
    return {"documents": n,
            "doc_mean_nll": sum(s["mean_nll"] for s in stats) / n,
            "token_mean_nll": sum(s["sum_nll"] for s in stats) / sum(s["tokens"] for s in stats),
            "relative_cost": sum(s["cost"] for s in stats) / sum(s["base16"] for s in stats),
            "depth_share": {d: sum(x == d for s in stats for x in s["depths"]) /
                            max(1, sum(len(s["depths"]) for s in stats)) for d in V3}}


def bootstrap(diff, repeats=10000, seed=20261012):
    x = torch.tensor(diff, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    m = x[torch.randint(0, len(x), (repeats, len(x)), generator=g)].mean(1)
    return {"mean": float(x.mean()), "sd": float(x.std()), "n": len(x),
            "ci95": [float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))],
            "lower_one_sided_95": float(torch.quantile(m, 0.05))}


def best_static(uniform, idx, budget):
    """Mixture of <= 2 uniform depths with realised relative cost <= budget, lowest doc-mean NLL."""
    nll = {d: sum(uniform[k][d]["mean_nll"] for k in idx) / len(idx) for d in V3}
    rel = {d: sum(uniform[k][d]["cost"] for k in idx) / sum(uniform[k][d]["base16"] for k in idx) for d in V3}
    best = None
    for i in V3:
        if rel[i] <= budget and (best is None or nll[i] < best[0]):
            best = (nll[i], [(i, 1.0)])
        for j in V3:
            if rel[i] <= budget < rel[j] and nll[j] < nll[i]:
                w = (budget - rel[i]) / (rel[j] - rel[i])
                q = (1 - w) * nll[i] + w * nll[j]
                if best is None or q < best[0]:
                    best = (q, [(i, 1 - w), (j, w)])
    return best[1]


# ----------------------------------------------------------------------------- KV coverage study

@torch.no_grad()
def kv_study(model, items, tables, idx, seed=20261012):
    g = torch.Generator().manual_seed(seed)
    tab_idx = {d: DEPTHS.index(d) for d in V3}
    rows = []
    for k in idx:
        item = items[k]
        _, lines, _ = lines_of(item)
        for _ in range(2):
            plan = [V3[int(i)] for i in torch.randint(0, 5, (len(lines),), generator=g)]
            nll, depths = retry(online, model, item, lambda z, dz, c, pl, plan=plan: plan[c - 1])
            hist = []
            for (a, b), d in zip(lines, depths):
                sc = scored_in(item, a, b)
                if sc and hist:
                    m_tot = sum(m for m, _ in hist)
                    short = sum(m * max(d - dj, 0) for m, dj in hist) / m_tot
                    excess = sum(m * max(dj - d, 0) for m, dj in hist) / m_tot
                    hmax = max(dj for _, dj in hist)
                    diff = float((nll[sc].double() - tables[k][tab_idx[d], sc].double()).mean())
                    rows.append({"d": d, "shortfall": short, "excess": excess, "delta_max": d - hmax,
                                 "diff": diff, "n": len(sc)})
                hist.append((b - a, d))
    x = torch.tensor([[1.0, r["shortfall"], r["excess"]] for r in rows], dtype=torch.float64)
    w = torch.tensor([r["n"] for r in rows], dtype=torch.float64)
    y = torch.tensor([r["diff"] for r in rows], dtype=torch.float64)
    beta = torch.linalg.solve(x.T @ (w[:, None] * x), x.T @ (w * y))
    resid = y - x @ beta
    groups = {
        "delta_max<=0_and_shortfall=0": [r for r in rows if r["delta_max"] <= 0 and r["shortfall"] == 0],
        "delta_max<=0_and_shortfall>0": [r for r in rows if r["delta_max"] <= 0 and r["shortfall"] > 0],
        "delta_max>0": [r for r in rows if r["delta_max"] > 0],
    }

    def summ(rs):
        if not rs:
            return None
        ww = torch.tensor([r["n"] for r in rs], dtype=torch.float64)
        yy = torch.tensor([r["diff"] for r in rs], dtype=torch.float64)
        return {"lines": len(rs), "token_weighted_mean": float((ww * yy).sum() / ww.sum()),
                "line_sd": float(yy.std()) if len(rs) > 1 else None}
    return {"lines": len(rows), "per_token_diff_model": {"intercept": float(beta[0]),
            "per_missing_iteration": float(beta[1]), "per_excess_iteration": float(beta[2]),
            "residual_sd": float(resid.std())},
            "groups": {k: summ(v) for k, v in groups.items()}}


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.5, 0.6])
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    blob = torch.load(OUT / "survey_tables.pt")
    items, tables = blob["items"], blob["tables"]
    model = H.load_model()
    log = {}
    t0 = time.perf_counter()

    xr, xp, y, groups, qx, qy, qg = fit_rows(model, items, tables, FIT)
    rep_model, rep_alpha = fit_ridge(torch.cat([xr, xp], 1), y, groups, (1e2, 1e3, 1e4, 1e5))
    pos_model, pos_alpha = fit_ridge(xp, y, groups, (0.01, 0.1, 1.0, 10.0, 100.0))
    q_model, q_alpha = fit_ridge(qx, qy, qg, (1e2, 1e3, 1e4, 1e5))
    log["fit"] = {"rows": len(y), "question_rows": len(qy), "alphas": {"rep": rep_alpha, "position": pos_alpha,
                  "question": q_alpha}, "seconds": time.perf_counter() - t0}
    print(json.dumps(log["fit"]), flush=True)

    def rep_policy(nu):
        def pol(z, dz, c, prev_len):
            onehot = torch.tensor([1.0 if dz == d else 0.0 for d in V3], dtype=torch.float64)
            x = torch.cat([z.double(), onehot, torch.tensor(pos_feats(c, prev_len), dtype=torch.float64)])[None]
            return V3[int((rep_model(x)[0] + nu * R_REL).argmin())]
        return pol

    def pos_policy(nu):
        def pol(z, dz, c, prev_len):
            x = torch.tensor([pos_feats(c, prev_len)], dtype=torch.float64)
            return V3[int((pos_model(x)[0] + nu * R_REL).argmin())]
        return pol

    # uniform runs (question at 16, all lines at d) for cal and test items
    t1 = time.perf_counter()
    uniform = {}
    for k in list(CAL) + list(TEST):
        uniform[k] = {}
        for d in V3:
            nll, depths = retry(online, model, items[k], lambda z, dz, c, pl, d=d: d)
            uniform[k][d] = doc_stats(items[k], nll, depths)
    log["uniform_seconds"] = time.perf_counter() - t1
    print("uniform done", round(log["uniform_seconds"]), flush=True)

    # question-level predictions (representation at question end from the uniform-16 prefix equals the
    # online question segment, so reuse a direct computation)
    q_pred = {}
    for k in list(CAL) + list(TEST):
        item = items[k]
        q_end, _, S = lines_of(item)
        ids = torch.tensor(item["ids"][:q_end], device="cuda")[None]
        state = state_matrix(item["row"], q_end).cuda()[None]
        cache = new_cache()
        z, _ = run_segment(model, cache, ids, state, 0, q_end, 16)
        q_pred[k] = q_model(z[-1].double().cpu()[None])[0]

    # one real-execution scan per policy on cal, shared by all budgets
    scans = {}
    for name, make in (("position", pos_policy), ("rep", rep_policy)):
        scans[name] = []
        for nu in NU_GRID:
            stats = []
            for k in CAL:
                nll, depths = retry(online, model, items[k], make(nu))
                stats.append(doc_stats(items[k], nll, depths))
            agg = aggregate(stats)
            scans[name].append({"nu": nu, "relative_cost": agg["relative_cost"], "doc_mean_nll": agg["doc_mean_nll"],
                                "depth_share": agg["depth_share"]})
            print(name, json.dumps(scans[name][-1]), flush=True)
    test_cache = {}

    def test_run(name, nu):
        if (name, nu) not in test_cache:
            make = pos_policy if name == "position" else rep_policy
            stats = []
            for k in TEST:
                nll, depths = retry(online, model, items[k], make(nu))
                stats.append(doc_stats(items[k], nll, depths))
            test_cache[(name, nu)] = stats
        return test_cache[(name, nu)]

    results = {"budgets": {}, "cal_scans": scans}
    base = sum(uniform[k][16]["base16"] for k in TEST)
    for B in args.budgets:
        res = {}
        support = best_static(uniform, CAL, B)

        def static_doc(k):
            return sum(w * uniform[k][d]["mean_nll"] for d, w in support)

        static_cost = sum(sum(w * uniform[k][d]["cost"] for d, w in support) for k in TEST)
        res["static"] = {"support": support, "test_doc_mean_nll": sum(static_doc(k) for k in TEST) / len(TEST),
                         "test_relative_cost": static_cost / base}

        best_q = None
        for nu in NU_GRID + (8.0, 12.0):
            pick = {k: V3[int((q_pred[k] + nu * R_REL).argmin())] for k in CAL}
            agg = aggregate([uniform[k][pick[k]] for k in CAL])
            if agg["relative_cost"] <= B and (best_q is None or agg["doc_mean_nll"] < best_q[1]):
                best_q = (nu, agg["doc_mean_nll"])
        if best_q is not None:
            pick = {k: V3[int((q_pred[k] + best_q[0] * R_REL).argmin())] for k in TEST}
            q_stats = [uniform[k][pick[k]] for k in TEST]
            res["question_level"] = {"nu": best_q[0], "test": aggregate(q_stats),
                                     "gain_vs_static": bootstrap([static_doc(k) - s["mean_nll"]
                                                                  for k, s in zip(TEST, q_stats)])}
        docs = {}
        for name in ("position", "rep"):
            feasible = [x for x in scans[name] if x["relative_cost"] <= B]
            if not feasible:
                res[name] = {"chosen_nu": None}
                continue
            chosen = min(feasible, key=lambda x: x["doc_mean_nll"])
            stats = test_run(name, chosen["nu"])
            docs[name] = {k: st["mean_nll"] for k, st in zip(TEST, stats)}
            res[name] = {"chosen_nu": chosen["nu"], "cal": chosen, "test": aggregate(stats),
                         "test_cost_over_static": sum(st["cost"] for st in stats) / static_cost,
                         "gain_vs_static": bootstrap([static_doc(k) - docs[name][k] for k in TEST])}
        if "rep" in docs and "position" in docs:
            res["rep_vs_position"] = bootstrap([docs["position"][k] - docs["rep"][k] for k in TEST])
            res["test_cost_ratio_rep_over_position"] = res["rep"]["test"]["relative_cost"] / \
                res["position"]["test"]["relative_cost"]
        results["budgets"][str(B)] = res
        print(json.dumps({B: res}, default=str), flush=True)

    t2 = time.perf_counter()
    results["kv_study"] = kv_study(model, items, tables, list(FIT)[:120])
    log["kv_seconds"] = time.perf_counter() - t2
    log["total_seconds"] = time.perf_counter() - t0
    write_json(RESULTS / "phase0_online.json", receipt(
        "chunk_depth_v3_1_phase0_online", split={"fit": [0, 299], "cal": [300, 419], "test": [420, 719]},
        depths=V3, relative_cost=R_REL.tolist(), nu_grid=NU_GRID, results=results, log=log))
    print(json.dumps(results["kv_study"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
