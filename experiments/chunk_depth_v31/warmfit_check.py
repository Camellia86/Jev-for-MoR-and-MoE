"""Diagnostic: is the v3.1 readout saturation produced by the warm-up objective or by its optimisation?

On the warm rows at the selected lambda (frozen model, as in train.warm_main):
  1. the saved readout: saturation share and mean objective, against the mean of the per-row optimum F*
     (objective_check: exact vertex test, numerical solve elsewhere);
  2. an exact replay of train.train_readout with per-step logging (first-step logit change, saturation);
  3. the same objective under three other optimisation settings: standardised inputs with the same AdamW
     (lr 0.003, 400 steps of 32), raw inputs with lr 3e-5, standardised inputs with full-batch L-BFGS.
Saturation = top mu within 1e-3 of the Dirichlet-mean ceiling; reported on the warm rows and on pilot_cal
decision rows (frozen representations). Writes results/chunk-depth-v31/warmfit_check.json.
"""

from __future__ import annotations

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v31.data import OUT, RESULTS
from experiments.chunk_depth_v31.objective_check import A, B, F, solve, vertex_optimal
from experiments.chunk_depth_v31.train import (K, TAU, WARM_SEED, aux_loss, doc_structure, frozen_boundary_reps,
                                               load_kv, mean_policy, new_readout, warm_rows)

CEIL = A + B


def objective(mu, c):
    uniform = torch.full_like(mu, 1.0 / K)
    return (mu * c).sum(-1) + aux_loss(mu, c, uniform)


def stats(lin, x, c, xcal, fstar):
    with torch.no_grad():
        mu = mean_policy(lin(x))
        mucal = mean_policy(lin(xcal))
        logits = lin(xcal).double()
    return {"train_saturated_share": float((mu.max(-1).values >= CEIL - 1e-3).double().mean()),
            "cal_saturated_share": float((mucal.max(-1).values >= CEIL - 1e-3).double().mean()),
            "train_mean_objective_minus_optimum": float(objective(mu, c).mean() - fstar),
            "cal_logit_range_median": float((logits.max(-1).values - logits.min(-1).values).median()),
            "weight_norm": float(lin.weight.norm())}


def adam_run(x, c, xcal, fstar, lr, probe):
    """train.train_readout with logging; x already on cuda (double)."""
    lin = new_readout().cuda().double()
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=0.01)
    g = torch.Generator(device="cpu").manual_seed(WARM_SEED)
    uniform = torch.full((K,), 1.0 / K, dtype=torch.float64, device="cuda")
    n, bs, pos = len(x), 32, 0
    order = torch.randperm(n, generator=g)
    trace, first_step = [], None
    for step in range(400):
        if pos + bs > n:
            order, pos = torch.randperm(n, generator=g), 0
        idx = order[pos:pos + bs].cuda()
        pos += bs
        with torch.no_grad():
            before = lin(x[probe])
        mu = mean_policy(lin(x[idx]))
        loss = ((mu * c[idx]).sum(-1) + aux_loss(mu, c[idx], uniform.expand_as(mu))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        with torch.no_grad():
            dl = (lin(x[probe]) - before).abs().max(-1).values
        if step == 0:
            first_step = {"median_max_abs_logit_change": float(dl.median()), "max": float(dl.max())}
        if step in (0, 1, 4, 9, 24, 49, 99, 199, 399):
            with torch.no_grad():
                sat = float((mean_policy(lin(x[probe])).max(-1).values >= CEIL - 1e-3).double().mean())
            trace.append({"step": step + 1, "probe_saturated_share": sat, "median_max_abs_logit_change": float(dl.median())})
    return lin, {"first_step": first_step, "trace": trace, **stats(lin, x, c, xcal, fstar)}


def lbfgs_run(x, c, xcal, fstar, l2=1e-4):
    lin = new_readout().cuda().double()
    opt = torch.optim.LBFGS(lin.parameters(), lr=1.0, max_iter=500, history_size=50, tolerance_grad=1e-9,
                            tolerance_change=1e-12, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        loss = objective(mean_policy(lin(x)), c).mean() + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss
    opt.step(closure)
    return {"l2": l2, **stats(lin, x, c, xcal, fstar)}


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    kappa_s, kappa_e = load_kv()
    blob = torch.load(OUT / "readouts.pt")
    lam = blob["selected_lambda"]
    items, feats = torch.load(OUT / "items_warm.pt"), torch.load(OUT / "features_warm.pt")
    cal, cal_feats = torch.load(OUT / "items_pilot_cal.pt"), torch.load(OUT / "features_pilot_cal.pt")
    model = H.load_model()
    reps = frozen_boundary_reps(model, items, feats)
    cal_reps = frozen_boundary_reps(model, cal, cal_feats)
    x, c = warm_rows(items, feats, reps, lam, kappa_s, kappa_e)
    del feats, cal_feats, model
    torch.cuda.empty_cache()
    xcal = torch.cat([torch.cat([z[k], torch.tensor([l["phi"] for l in doc_structure(it)])], 1)
                      for it, z in zip(cal, cal_reps) for k in range(K)]).double()
    # per-row optimum of the objective
    v = vertex_optimal(c)
    mu_star = torch.full_like(c, B)
    mu_star[v >= 0] = B
    mu_star[torch.arange(len(c))[v >= 0], v[v >= 0]] = A + B
    if (v < 0).any():
        mu_star[v < 0] = solve(c[v < 0])
    s = torch.softmax(-c / TAU, -1)
    fstar = float(F(mu_star, c, s).mean())
    out = {"lambda": lam, "rows": len(x), "cal_rows": len(xcal), "mean_optimum_objective": fstar,
           "optimum_saturated_share": float((mu_star.max(-1).values >= CEIL - 1e-3).double().mean())}
    x, c, xcal = x.cuda(), c.cuda(), xcal.cuda()
    saved = new_readout()
    saved.load_state_dict(blob["readouts"][lam])
    saved = saved.cuda().double()
    out["saved_readout"] = stats(saved, x, c, xcal, fstar)
    probe = torch.arange(0, len(x), max(1, len(x) // 512), device="cuda")
    replay, out["replay_raw_lr3e-3"] = adam_run(x, c, xcal, fstar, 0.003, probe)
    out["replay_matches_saved_max_abs_weight_diff"] = float((replay.weight - saved.weight).abs().max())
    mu_, sd_ = x[:, :5280].mean(0), x[:, :5280].std(0) + 1e-6
    xs = torch.cat([(x[:, :5280] - mu_) / sd_, x[:, 5280:]], 1)
    xcals = torch.cat([(xcal[:, :5280] - mu_) / sd_, xcal[:, 5280:]], 1)
    _, out["standardised_lr3e-3"] = adam_run(xs, c, xcals, fstar, 0.003, probe)
    _, out["raw_lr3e-5"] = adam_run(x, c, xcal, fstar, 3e-5, probe)
    out["standardised_lbfgs"] = lbfgs_run(xs, c, xcals, fstar)
    write_json(RESULTS / "warmfit_check.json", receipt("chunk_depth_v3_1_warm_fit_check", **out))
    for k in ("saved_readout", "replay_raw_lr3e-3", "standardised_lr3e-3", "raw_lr3e-5", "standardised_lbfgs"):
        v = out[k]
        print(k, {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != "trace"})
    print({k: out[k] for k in ("mean_optimum_objective", "optimum_saturated_share", "replay_matches_saved_max_abs_weight_diff")})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
