"""Policy-side mathematics for protocol jev-shared-depth-v2 (sections 4.2, 6.2, 6.3).

All actor quantities are float64. Costs passed to route surrogates are detached.
"""

from __future__ import annotations

import math

import torch

KAPPA, EPSILON, ETA, ACTIONS = 256.0, 0.01, 0.001, 4
TAU, CE_W, BRIER_W, KL_W = 0.05, 0.1, 0.01, 0.1
SAMPLES = 64


def dirichlet_alpha(logits: torch.Tensor) -> torch.Tensor:
    p = torch.softmax(logits.double(), -1)
    return KAPPA * ((1 - EPSILON) * p + EPSILON / ACTIONS) + ETA


def mean_policy(logits: torch.Tensor) -> torch.Tensor:
    alpha = dirichlet_alpha(logits)
    return alpha / alpha.sum(-1, keepdim=True)


def soft_label(cost: torch.Tensor) -> torch.Tensor:
    return torch.softmax(-cost.detach().double() / TAU, -1)


def auxiliary(mu: torch.Tensor, cost: torch.Tensor, mu_ref: torch.Tensor) -> torch.Tensor:
    """0.1 CE(s, mu) + 0.01 Brier + 0.1 KL(mu || mu_ref); s is a stop-gradient soft label."""
    s = soft_label(cost)
    ce = -(s * mu.log()).sum(-1)
    brier = ((mu - s) ** 2).sum(-1)
    kl = (mu * (mu.log() - mu_ref.double().log())).sum(-1)
    return CE_W * ce + BRIER_W * brier + KL_W * kl


def route_exact(mu: torch.Tensor, cost: torch.Tensor) -> torch.Tensor:
    return (mu * cost.detach().double()).sum(-1)


def dirichlet_log_prob(q: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    return (torch.lgamma(alpha.sum(-1)) - torch.lgamma(alpha).sum(-1)
            + ((alpha - 1) * q.log()).sum(-1))


def sample_dirichlet(alpha: torch.Tensor, m: int, seed: int) -> torch.Tensor:
    """Deterministic CPU float64 Dirichlet samples [m, 4] for a detached alpha [4]."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = alpha.detach().double().cpu().expand(m, -1).contiguous()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(int(torch.randint(0, 2**62, (1,), generator=g)))
        gam = torch._standard_gamma(a)
    q = gam / gam.sum(-1, keepdim=True)
    return q.clamp_min(1e-300)


def route_jev(alpha: torch.Tensor, cost: torch.Tensor, seed: int, m: int = SAMPLES) -> tuple[torch.Tensor, dict]:
    """Leave-one-out score-function surrogate; q, v and baselines are detached."""
    q = sample_dirichlet(alpha, m, seed).to(alpha.device)
    v = q @ cost.detach().double()
    baseline = (v.sum() - v) / (m - 1)
    advantage = (v - baseline).detach()
    surrogate = (advantage * dirichlet_log_prob(q, alpha)).mean()
    return surrogate, {"v_mean": float(v.mean()), "v_sd": float(v.std())}


def tilt(mu: torch.Tensor, f: torch.Tensor, b: float) -> torch.Tensor:
    logw = mu.double().log() - b * f.double()
    return torch.softmax(logw, -1)


def mean_cost(mu: torch.Tensor, f: torch.Tensor, b: float) -> float:
    return float((tilt(mu, f, b) * f.double()).sum(-1).mean())


def calibrate(mu: torch.Tensor, f: torch.Tensor, target: float, tol: float = 1e-4) -> dict:
    """Bisection on b so that the mean expected cost equals target (section 6.3)."""
    lo, hi = -500.0, 500.0
    c_lo, c_hi = mean_cost(mu, f, lo), mean_cost(mu, f, hi)
    if target >= c_lo:
        return {"b": lo, "cost": c_lo, "boundary": "max_cost"}
    if target <= c_hi:
        return {"b": hi, "cost": c_hi, "boundary": "min_cost"}
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        c = mean_cost(mu, f, mid)
        if abs(c - target) <= tol * 0.01:
            break
        if c > target:
            lo = mid
        else:
            hi = mid
    c = mean_cost(mu, f, mid)
    if abs(c - target) > tol:
        raise RuntimeError(f"calibration failed: {c} vs {target}")
    return {"b": mid, "cost": c, "boundary": None}


def best_static_mixture(points: list[dict], budget: float = 1.0) -> dict:
    """Minimum-NLL convex combination of at most two candidate points with cost <= budget.

    points: [{"name", "cost", "nll"}] in pool order. Tie-break: lower cost, fewer support
    points, earlier pool order.
    """
    best = None

    def consider(entry):
        nonlocal best
        key = (round(entry["nll"], 12), round(entry["cost"], 12), len(entry["support"]), entry["order"])
        if best is None or key < best[0]:
            best = (key, entry)

    for i, p in enumerate(points):
        if p["cost"] <= budget + 1e-12:
            consider({"support": [(p["name"], 1.0)], "cost": p["cost"], "nll": p["nll"], "order": (i,)})
    for i, p in enumerate(points):
        for j, q in enumerate(points):
            if p["cost"] <= budget < q["cost"] and q["nll"] < p["nll"]:
                w = (budget - p["cost"]) / (q["cost"] - p["cost"])
                consider({"support": [(p["name"], 1 - w), (q["name"], w)],
                          "cost": (1 - w) * p["cost"] + w * q["cost"],
                          "nll": (1 - w) * p["nll"] + w * q["nll"], "order": (min(i, j), max(i, j))})
    if best is None:
        raise ValueError("no feasible static point")
    return best[1]


def bootstrap_mean(x: torch.Tensor, repeats: int = 10000, seed: int = 20261011) -> dict:
    x = x.double().cpu()
    g = torch.Generator(device="cpu").manual_seed(seed)
    n = x.numel()
    idx = torch.randint(0, n, (repeats, n), generator=g)
    means = x[idx].mean(1)
    centered = means - x.mean()
    return {
        "mean": float(x.mean()), "sd": float(x.std()), "n": n,
        "ci95": [float(torch.quantile(means, 0.025)), float(torch.quantile(means, 0.975))],
        "lower_one_sided_95": float(torch.quantile(means, 0.05)),
        "upper_one_sided_95": float(torch.quantile(means, 0.95)),
        "p_one_sided_greater": float((centered >= x.mean()).double().mean()),
        "p_two_sided": float((centered.abs() >= abs(float(x.mean()))).double().mean()),
    }


def lr_at(update: int, total: int, peak: float = 1e-4, end: float = 1e-5, warmup_frac: float = 0.05) -> float:
    warm = max(1, int(round(total * warmup_frac)))
    if update < warm:
        return peak * (update + 1) / warm
    progress = (update - warm) / max(1, total - warm)
    return end + 0.5 * (peak - end) * (1 + math.cos(math.pi * progress))
