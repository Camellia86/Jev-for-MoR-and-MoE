"""Diagnostic for the v3.1 warm-up objective (train.py, train_readout): where is its per-row optimum?

For one row with per-token cost vector c, the warm-up loss as a function of mu is
  F(mu) = c.mu - 0.1 sum_k s_k log mu_k + 0.01 sum_k (mu_k - s_k)^2 + 0.1 sum_k mu_k log(K mu_k),
  s = softmax(-c / tau), mu = a p + b, p in the simplex,
with a = kappa (1 - eps) / (kappa + K eta), b = (kappa eps / K + eta) / (kappa + K eta). F is convex in mu, so
the vertex mu = a e_t + b is the global minimiser iff every direction e_j - e_t (j != t) has a positive
directional derivative. The check is exact; rows whose optimum is not a vertex are solved numerically.
Rows are the warm-up rows of train.warm_rows (cost only; no representation is needed).
Writes results/chunk-depth-v31/objective_check.json.
"""

from __future__ import annotations

import math

import torch

from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v31.data import OUT, RESULTS
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31.train import EPS, ETA, K, KAPPA_D, LAMBDA_GRID, TAU, doc_structure, load_kv

A = KAPPA_D * (1 - EPS) / (KAPPA_D + K * ETA)
B = (KAPPA_D * EPS / K + ETA) / (KAPPA_D + K * ETA)


def grad_F(mu, c, s):
    return c - 0.1 * s / mu + 0.02 * (mu - s) + 0.1 * (torch.log(K * mu) + 1)


def F(mu, c, s):
    return (c * mu).sum(-1) - 0.1 * (s * mu.log()).sum(-1) + 0.01 * ((mu - s) ** 2).sum(-1) \
        + 0.1 * (mu * torch.log(K * mu)).sum(-1)


def vertex_optimal(c):
    """[N] index of the optimal vertex, or -1 if no vertex is optimal."""
    s = torch.softmax(-c / TAU, -1)
    out = torch.full((len(c),), -1, dtype=torch.long)
    for t in range(K):
        mu = torch.full_like(c, B)
        mu[:, t] = A + B
        g = grad_F(mu, c, s)
        d = g - g[:, t:t + 1]                       # directional derivatives along e_j - e_t
        d[:, t] = 1.0
        ok = (d > 0).all(-1)
        out[ok] = t
    return out


def solve(c, steps=4000):
    """Numerical minimiser over logits (float64, Adam); used for rows without a vertex optimum."""
    s = torch.softmax(-c / TAU, -1)
    logits = (-c / TAU).clone().requires_grad_(True)
    opt = torch.optim.Adam([logits], lr=0.05)
    for _ in range(steps):
        mu = A * torch.softmax(logits, -1) + B
        loss = F(mu, c, s).sum()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        return A * torch.softmax(logits, -1) + B


def two_level_threshold():
    """Smallest gap D such that c = (0, D, D, D, D) has its optimum at the vertex of action 0."""
    lo, hi = 0.0, 2.0
    for _ in range(60):
        mid = (lo + hi) / 2
        c = torch.tensor([[0.0] + [mid] * (K - 1)], dtype=torch.float64)
        if vertex_optimal(c)[0] == 0:
            hi = mid
        else:
            lo = mid
    return hi


def warm_costs(items, feats, lam, kappa_s, kappa_e):
    rows = []
    for item, f in zip(items, feats):
        st = doc_structure(item)
        for i, l in enumerate(st):
            if l["n"] == 0:
                continue
            L = f["nll"][:, l["scored"]].double().sum(1)
            for dprev in (R.V3 if i > 0 else (16,)):
                hist = [(st[j]["b"] - st[j]["a"], dprev) for j in range(i)]
                kv = torch.tensor([kappa_s * R.coverage(hist, d)[0] + kappa_e * R.coverage(hist, d)[1] for d in R.V3],
                                  dtype=torch.float64)
                rows.append((L + lam * l["n"] * R.R_REL + l["n"] * kv) / l["n"])
    return torch.stack(rows)


def main() -> int:
    kappa_s, kappa_e = load_kv()
    items = torch.load(OUT / "items_warm.pt")
    feats = torch.load(OUT / "features_warm.pt")
    ceiling = A + B
    out = {"a": A, "b": B, "log_ceiling_over_floor": math.log((A + B) / B),
           "example_c_t0_cj0.6": None, "two_level_gap_threshold": two_level_threshold(), "lambdas": {}}
    c = torch.tensor([[0.0, 0.6, 0.6, 0.6, 0.6]], dtype=torch.float64)
    s = torch.softmax(-c / TAU, -1)
    mu = torch.full_like(c, B)
    mu[0, 0] = A + B
    g = grad_F(mu, c, s)[0]
    out["example_c_t0_cj0.6"] = {"directional_derivative": float(g[1] - g[0]), "vertex_optimal": int(vertex_optimal(c)[0])}
    for lam in LAMBDA_GRID:
        c = warm_costs(items, feats, lam, kappa_s, kappa_e)
        v = vertex_optimal(c)
        best = c.argmin(-1)
        sorted_c = c.sort(-1).values
        gap = sorted_c[:, 1] - sorted_c[:, 0]
        interior = v < 0
        mu_int = solve(c[interior]) if interior.any() else torch.empty(0, K, dtype=torch.float64)
        top_int = mu_int.max(-1).values
        out["lambdas"][str(lam)] = {
            "rows": len(c),
            "vertex_optimum_share": float((v >= 0).double().mean()),
            "vertex_is_cost_argmin_share": float((v[v >= 0] == best[v >= 0]).double().mean()) if (v >= 0).any() else None,
            "gap_best_to_second_quantiles": [float(x) for x in torch.quantile(gap, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))],
            "non_vertex_rows_top_mu_quantiles": [float(x) for x in torch.quantile(top_int, torch.tensor([0.1, 0.5, 0.9], dtype=torch.float64))]
            if len(top_int) else None,
            "non_vertex_rows_within_1e-3_of_ceiling": float((top_int >= ceiling - 1e-3).double().mean()) if len(top_int) else None,
            "argmin_share": [float((best == k).double().mean()) for k in range(K)],
        }
        print(lam, out["lambdas"][str(lam)], flush=True)
    write_json(RESULTS / "objective_check.json", receipt("chunk_depth_v3_1_warm_objective_check", **out))
    print({k: out[k] for k in ("a", "b", "log_ceiling_over_floor", "two_level_gap_threshold", "example_c_t0_cj0.6")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
