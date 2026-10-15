"""Stage 1 step 2 (plan v3.2 sections 4.2-4.4, 4.6): cross-entropy readout, position schedule R and the
representation-ridge diagnostic, all fitted on readout_train tables under theta_S.

Readout: p = softmax(W x + beta), x = [(z - m) / sigma ; phi]; objective = weighted soft-label cross entropy
on p + gamma ||W||_F^2, convex, solved by full-batch float64 L-BFGS from zero. gamma by 5-fold grouped CV at
lambda 1.5; lambda by the greedy b = 0 table rule. Writes artifacts/chunk-depth-v32/{head,R,rep_ridge}.pt and
results/chunk-depth-v32/head.json.
"""

from __future__ import annotations

import json
import math
import time

import torch

from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v3.phase0_online import Ridge, fit_ridge
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31.train import K, doc_structure, load_kv
from experiments.chunk_depth_v32.common import OUT, RESULTS, load_role

TAU = 0.05
BUDGET = 0.5
LAMBDA_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)
GAMMA_GRID = (1e-6, 1e-5, 1e-4, 1e-3, 1e-2)
CV_LAMBDA = 1.5


def build_rows(items, tabs, kappa_s, kappa_e):
    """One row per (line with scored tokens, previous depth): line 1 only after the depth-16 question."""
    z, phi, lbar, kv, w, doc, kprev = [], [], [], [], [], [], []
    for j, (item, t) in enumerate(zip(items, tabs)):
        st = doc_structure(item)
        ntot = sum(l["n"] for l in st)
        for i, l in enumerate(st):
            if l["n"] == 0:
                continue
            prevs = R.V3 if i > 0 else (16,)
            for dprev in prevs:
                k = R.V3.index(dprev)
                hist = [(st[q]["b"] - st[q]["a"], dprev) for q in range(i)]
                cov = [R.coverage(hist, d) for d in R.V3]
                z.append(t["reps"][k, i])
                phi.append(l["phi"])
                lbar.append(t["line_nll"][i])
                kv.append([kappa_s * s + kappa_e * e for s, e in cov])
                w.append(l["n"] / ntot / len(prevs))
                doc.append(j)
                kprev.append(k)
    return {"z": torch.stack(z), "phi": torch.tensor(phi, dtype=torch.float64), "lbar": torch.stack(lbar),
            "kv": torch.tensor(kv, dtype=torch.float64), "w": torch.tensor(w, dtype=torch.float64),
            "doc": torch.tensor(doc), "kprev": torch.tensor(kprev)}


def soft_labels(rows, lam):
    cost = rows["lbar"] + lam * R.R_REL + rows["kv"]
    return torch.softmax(-cost / TAU, -1), cost


def fit(X, S, w, gamma, max_iter=2000):
    W = torch.zeros(K, X.shape[1], dtype=torch.float64, device=X.device, requires_grad=True)
    beta = torch.zeros(K, dtype=torch.float64, device=X.device, requires_grad=True)
    wn = w / w.sum()
    opt = torch.optim.LBFGS([W, beta], lr=1.0, max_iter=max_iter, max_eval=int(max_iter * 1.25), history_size=50,
                            tolerance_grad=1e-7, tolerance_change=1e-14, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        logp = torch.log_softmax(X @ W.T + beta, -1)
        loss = -(wn * (S * logp).sum(-1)).sum() + gamma * (W ** 2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    loss = closure()
    grad_inf = float(torch.cat([W.grad.flatten(), beta.grad]).abs().max())
    n_iter = opt.state[opt._params[0]]["n_iter"]
    return W.detach(), beta.detach(), {"iterations": int(n_iter), "grad_inf": grad_inf, "objective": float(loss.detach())}


def weighted_ce(X, S, w, W, beta):
    logp = torch.log_softmax(X @ W.T + beta, -1)
    return float((w * -(S * logp).sum(-1)).sum() / w.sum())


def greedy_cost(items, tabs, W, beta, m, sd):
    """b = 0 sequential table execution: the previous depth is the chosen one, the representation its table run."""
    num = den = 0.0
    for item, t in zip(items, tabs):
        st = doc_structure(item)
        kprev = R.V3.index(16)
        for i, l in enumerate(st):
            z = t["reps"][kprev, i].double().to(W.device)
            x = torch.cat([(z - m) / sd, torch.tensor(l["phi"], dtype=torch.float64, device=W.device)])
            k = int((W @ x + beta).argmax())
            num += float(l["f"][k])
            den += float(l["f"][R.V3.index(16)])
            kprev = k
    return num / den


class DeviceRidge(Ridge):
    """phase0_online.Ridge with the intercept column and the penalty matrix on the data's device."""

    def __init__(self, x, y, alpha):
        self.mu, self.sd = x.mean(0), x.std(0) + 1e-6
        a = torch.cat([(x - self.mu) / self.sd, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], 1)
        eye = torch.eye(a.shape[1], dtype=x.dtype, device=x.device)
        eye[-1, -1] = 0
        self.w = torch.linalg.solve(a.T @ a + alpha * eye, a.T @ y)

    def __call__(self, x):
        a = torch.cat([(x - self.mu) / self.sd, torch.ones(len(x), 1, dtype=x.dtype, device=x.device)], 1)
        return a @ self.w


def fit_ridge_device(x, y, groups, alphas):
    """phase0_online.fit_ridge (5-fold grouped CV on squared error, refit on all rows) with DeviceRidge."""
    best = None
    for alpha in alphas:
        err = 0.0
        for f in range(5):
            tr, te = groups % 5 != f, groups % 5 == f
            err += float(((DeviceRidge(x[tr], y[tr], alpha)(x[te]) - y[te]) ** 2).sum())
        if best is None or err < best[1]:
            best = (alpha, err)
    return DeviceRidge(x, y, best[0]), best[0]


def ridge_state(r):
    return {"w": r.w.cpu(), "mu": r.mu.cpu(), "sd": r.sd.cpu()}


def ridge_from_state(s):
    r = Ridge.__new__(Ridge)
    r.w, r.mu, r.sd = s["w"], s["mu"], s["sd"]
    return r


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    t_start = time.perf_counter()
    kappa_s, kappa_e = load_kv()
    items = load_role("readout_train")
    tabs = torch.load(OUT / "tables_readout_train.pt")
    rows = build_rows(items, tabs, kappa_s, kappa_e)
    z = rows["z"].double()
    m, sd = z.mean(0), z.std(0) + 1e-6
    X = torch.cat([(z - m) / sd, rows["phi"]], 1).cuda()
    w, doc = rows["w"].cuda(), rows["doc"].cuda()
    out = {"rows": len(X), "documents": len(items), "tau": TAU, "gamma_grid": GAMMA_GRID, "lambda_grid": LAMBDA_GRID}
    print("rows", len(X), flush=True)

    # gamma by 5-fold grouped CV at lambda 1.5
    S, _ = soft_labels(rows, CV_LAMBDA)
    S = S.cuda()
    cv = {}
    for gamma in GAMMA_GRID:
        num = den = 0.0
        fits = []
        for f in range(5):
            tr, te = doc % 5 != f, doc % 5 == f
            W, beta, info = fit(X[tr], S[tr], w[tr], gamma)
            ce = weighted_ce(X[te], S[te], w[te], W, beta)
            num += ce * float(w[te].sum())
            den += float(w[te].sum())
            fits.append(info)
        cv[gamma] = {"held_out_ce": num / den, "max_grad_inf": max(i["grad_inf"] for i in fits),
                     "max_iterations": max(i["iterations"] for i in fits)}
        print("gamma", gamma, cv[gamma], flush=True)
    gamma = min(cv, key=lambda g: (cv[g]["held_out_ce"], g))
    entropy = float((w * -(S * S.clamp_min(1e-300).log()).sum(-1)).sum() / w.sum())
    out["cv"] = {"lambda": CV_LAMBDA, "by_gamma": {str(g): v for g, v in cv.items()}, "selected_gamma": gamma,
                 "label_entropy": entropy}

    # lambda rule
    m_c, sd_c = m.cuda(), sd.cuda()
    heads, lam_rows = {}, []
    for lam in LAMBDA_GRID:
        S, _ = soft_labels(rows, lam)
        S = S.cuda()
        W, beta, info = fit(X, S, w, gamma)
        b = greedy_cost(items, tabs, W, beta, m_c, sd_c)
        with torch.no_grad():
            p = torch.softmax(X @ W.T + beta, -1)
            kl = float((w * (S * (S.clamp_min(1e-300).log() - p.log())).sum(-1)).sum() / w.sum())
            sharp = {"share_max_p_ge_0.999": float((p.max(-1).values >= 0.999).double().mean()),
                     "share_max_s_ge_0.999": float((S.max(-1).values >= 0.999).double().mean()),
                     "median_max_p": float(p.max(-1).values.median()), "median_max_s": float(S.max(-1).values.median())}
        heads[lam] = {"W": W.cpu(), "beta": beta.cpu()}
        lam_rows.append({"lambda": lam, "greedy_relative_cost": b, "abs_log_ratio": abs(math.log(b / BUDGET)),
                         "fit": info, "train_kl_s_p": kl, **sharp})
        print(json.dumps(lam_rows[-1]), flush=True)
    best = min(lam_rows, key=lambda r: (round(r["abs_log_ratio"], 12), r["lambda"]))
    lam = best["lambda"]
    out["lambda_rule"] = {"rows": lam_rows, "selected_lambda": lam}
    torch.save({"W": heads[lam]["W"], "beta": heads[lam]["beta"], "m": m, "sd": sd, "lambda": lam, "gamma": gamma,
                "tau": TAU, "all_lambdas": heads}, OUT / "head.pt")
    del X, S
    torch.cuda.empty_cache()

    # comparator R: position ridge on line NLL (one row per line), grouped CV alpha
    px, py, pg = [], [], []
    for j, (item, t) in enumerate(zip(items, tabs)):
        for i, l in enumerate(doc_structure(item)):
            if l["n"]:
                px.append(l["phi"]); py.append(t["line_nll"][i]); pg.append(j)
    ridge, alpha = fit_ridge(torch.tensor(px, dtype=torch.float64), torch.stack(py), torch.tensor(pg),
                             (0.01, 0.1, 1.0, 10.0, 100.0))
    torch.save({**ridge_state(ridge), "alpha": alpha}, OUT / "R.pt")
    out["R"] = {"lines": len(px), "alpha": alpha}
    print("R alpha", alpha, flush=True)

    out["rep_ridge"] = rep_ridge(rows, z)
    out["seconds"] = time.perf_counter() - t_start
    write_json(RESULTS / "head.json", receipt("chunk_depth_v3_2_head", **out))
    return 0


def rep_ridge(rows, z):
    """Diagnostic: representation ridge (phase-0 recipe) on the same rows, raw z + previous-depth one-hot + phi."""
    onehot = torch.nn.functional.one_hot(rows["kprev"], K).double()
    XR = torch.cat([z, onehot, rows["phi"]], 1).cuda()
    probe, alpha = fit_ridge_device(XR, rows["lbar"].cuda(), rows["doc"].cuda(), (1e2, 1e3, 1e4, 1e5))
    torch.save({**ridge_state(probe), "alpha": alpha}, OUT / "rep_ridge.pt")
    print("rep ridge alpha", alpha, flush=True)
    return {"rows": len(XR), "alpha": alpha}


def finish(log_path):
    """Complete a run that stopped after head.pt and R.pt were saved: CV and lambda rows are taken from the
    run's log (exact prints of the computed values); the selected head's gradient is re-evaluated at the
    saved parameters; the representation ridge is fitted; head.json is written."""
    import ast
    t_start = time.perf_counter()
    kappa_s, kappa_e = load_kv()
    items = load_role("readout_train")
    tabs = torch.load(OUT / "tables_readout_train.pt")
    rows = build_rows(items, tabs, kappa_s, kappa_e)
    blob = torch.load(OUT / "head.pt")
    cv, lam_rows = {}, []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("gamma "):
            _, g, rest = line.split(" ", 2)
            cv[float(g)] = ast.literal_eval(rest)
        elif line.startswith('{"lambda"'):
            lam_rows.append(json.loads(line))
    gamma = min(cv, key=lambda g: (cv[g]["held_out_ce"], g))
    lam = min(lam_rows, key=lambda r: (round(r["abs_log_ratio"], 12), r["lambda"]))["lambda"]
    assert gamma == blob["gamma"] and lam == blob["lambda"], (gamma, lam, blob["gamma"], blob["lambda"])
    z = rows["z"].double()
    m, sd = z.mean(0), z.std(0) + 1e-6
    assert torch.equal(m, blob["m"]) and torch.equal(sd, blob["sd"])
    X = torch.cat([(z - m) / sd, rows["phi"]], 1).cuda()
    w = rows["w"].cuda()
    S, _ = soft_labels(rows, CV_LAMBDA)
    S = S.cuda()
    entropy = float((w * -(S * S.clamp_min(1e-300).log()).sum(-1)).sum() / w.sum())
    S, _ = soft_labels(rows, lam)
    S = S.cuda()
    W = blob["W"].cuda().requires_grad_(True)
    beta = blob["beta"].cuda().requires_grad_(True)
    logp = torch.log_softmax(X @ W.T + beta, -1)
    loss = -((w / w.sum()) * (S * logp).sum(-1)).sum() + gamma * (W ** 2).sum()
    loss.backward()
    recheck = {"objective": float(loss.detach()), "grad_inf": float(torch.cat([W.grad.flatten(), beta.grad]).abs().max())}
    del X, S, W, beta, logp, loss
    torch.cuda.empty_cache()
    out = {"rows": len(rows["w"]), "documents": len(items), "tau": TAU, "gamma_grid": GAMMA_GRID,
           "lambda_grid": LAMBDA_GRID,
           "cv": {"lambda": CV_LAMBDA, "by_gamma": {str(g): v for g, v in sorted(cv.items())}, "selected_gamma": gamma,
                  "label_entropy": entropy},
           "lambda_rule": {"rows": lam_rows, "selected_lambda": lam},
           "selected_head_recheck": recheck,
           "R": {"alpha": torch.load(OUT / "R.pt")["alpha"]},
           "provenance": f"cv and lambda rows parsed from {log_path.name}, the run that saved head.pt and R.pt; "
                         "it stopped at the representation ridge (device error), which finish mode fitted"}
    out["rep_ridge"] = rep_ridge(rows, z)
    out["finish_seconds"] = time.perf_counter() - t_start
    write_json(RESULTS / "head.json", receipt("chunk_depth_v3_2_head", **out))
    print(json.dumps({k: out[k] for k in ("selected_head_recheck", "R", "rep_ridge")}))
    return 0


if __name__ == "__main__":
    import sys
    from experiments.shared_depth_v2.common import ROOT
    if len(sys.argv) > 1 and sys.argv[1] == "--finish":
        raise SystemExit(finish(ROOT / "artifacts" / "v32_head.log"))
    raise SystemExit(main())
