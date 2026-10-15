"""Fitting and locking steps of v3.3 (plan sections 4.2, 4.4, 5.3). Written before any confirm-set online run.

    python -m experiments.chunk_depth_v33.lock A_fit --nshards n        (uses the GPU for the H ridge)
    python -m experiments.chunk_depth_v33.lock B_deploy_lock
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from experiments.shared_depth_v2.common import write_json
from experiments.chunk_depth_v31.train import K, doc_structure
from experiments.chunk_depth_v32.head import DeviceRidge, ridge_state
from experiments.chunk_depth_v32.stage1 import Head
from experiments.chunk_depth_v33 import common as C
from experiments.chunk_depth_v33 import policies as P

R_ALPHAS = (1e-2, 1e-1, 1.0, 10.0, 100.0, 1e3, 1e4)
H_ALPHAS = (1e3, 1e4, 1e5, 1e6, 1e7)
TARGETS = {0.5: (0.470, 0.485, 0.495, 0.505, 0.515, 0.530), 0.6: (0.570, 0.585, 0.595, 0.605, 0.615, 0.630)}
NU_GRID = torch.arange(0, 6.0001, 0.005, dtype=torch.float64)
B_GRID = torch.arange(-20, 20.0001, 0.05, dtype=torch.float64)
B_TARGETS = (0.480, 0.495, 0.505, 0.520)


def cv_ridge(x, y, groups, alphas):
    """5-fold cross-validation grouped by document on squared error; refit on all rows with the best alpha."""
    errs = {}
    for a in alphas:
        err = 0.0
        for f in range(5):
            tr, te = groups % 5 != f, groups % 5 == f
            err += float(((DeviceRidge(x[tr], y[tr], a)(x[te]) - y[te]) ** 2).sum())
        errs[a] = err
    best = min(alphas, key=lambda a: (errs[a], a))
    return DeviceRidge(x, y, best), best, {str(a): v for a, v in errs.items()}


def a_fit(nshards: int) -> int:
    t0 = time.perf_counter()
    kappa = P.kappas()
    fit_items, cal_items = C.load_role("fit"), C.load_role("calibration")
    fit_tabs = C.load_shards("tables_fit", nshards, scratch=True)
    cal_tabs = C.load_shards("tables_calibration", nshards, scratch=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    # R: one row per scored line, position features -> per-token table NLL at each depth
    px, py, pg = [], [], []
    hx, hy, hg = [], [], []
    for j, (item, t) in enumerate(zip(fit_items, fit_tabs)):
        for i, l in enumerate(doc_structure(item)):
            if not l["n"]:
                continue
            px.append(l["phi"]); py.append(t["line_nll"][i]); pg.append(j)
            for kprev in (range(K) if i > 0 else (4,)):
                hx.append(P.h_features(t["reps"][kprev, i], kprev, l["phi"]))
                hy.append(t["line_nll"][i]); hg.append(j)
    r_model, r_alpha, r_errs = cv_ridge(torch.tensor(px, dtype=torch.float64), torch.stack(py).double(),
                                        torch.tensor(pg), R_ALPHAS)
    X = torch.stack(hx).to(dev)
    del hx
    h_model, h_alpha, h_errs = cv_ridge(X, torch.stack(hy).double().to(dev), torch.tensor(hg, device=dev), H_ALPHAS)
    del X
    (C.ART / "fit").mkdir(parents=True, exist_ok=True)
    C.save_atomic({**ridge_state(r_model), "alpha": r_alpha}, C.ART / "fit" / "R.pt")
    C.save_atomic({**ridge_state(h_model), "alpha": h_alpha}, C.ART / "fit" / "H.pt")
    Rm = P.Ridge(torch.load(C.ART / "fit" / "R.pt", weights_only=False))
    Hm = P.Ridge(torch.load(C.ART / "fit" / "H.pt", weights_only=False))

    # cost prediction on the calibration set and parameter choice
    r_scores = [P.doc_scores_R(it, Rm) for it in cal_items]
    h_scores = [P.doc_scores_H(it, t, Hm) for it, t in zip(cal_items, cal_tabs)]
    pred = {"R": P.predicted_cost(r_scores, NU_GRID, kappa), "H": P.predicted_cost(h_scores, NU_GRID, kappa)}
    policies = {fam: {str(b): P.choose_parameters(NU_GRID, pred[fam], TARGETS[b]) for b in C.BUDGETS} for fam in pred}
    curve = {fam: [[round(float(NU_GRID[j]), 4), float(pred[fam][j])] for j in range(0, len(NU_GRID), 20)] for fam in pred}
    lock = C.v33_receipt(
        "v33_A_lock", fit_documents=len(fit_items), fit_rows={"R": len(px), "H": len(hy)},
        calibration_documents=len(cal_items), kappa={"kappa_s": kappa[0], "kappa_e": kappa[1]},
        R={"alpha": r_alpha, "alpha_grid": R_ALPHAS, "cv_squared_error": r_errs, "at_grid_edge": r_alpha == R_ALPHAS[-1],
           "sha256": C.sha256_file(C.ART / "fit" / "R.pt")},
        H={"alpha": h_alpha, "alpha_grid": H_ALPHAS, "cv_squared_error": h_errs, "at_grid_edge": h_alpha == H_ALPHAS[-1],
           "sha256": C.sha256_file(C.ART / "fit" / "H.pt")},
        nu_grid={"from": 0, "to": 6, "step": 0.005}, targets={str(k): v for k, v in TARGETS.items()},
        policies=policies, predicted_curve_every_0_1=curve, seconds=time.perf_counter() - t0,
        note="locked before any confirm-set online execution")
    write_json(C.RESULTS / "policies_A.json", lock)
    print(json.dumps({"R_alpha": r_alpha, "H_alpha": h_alpha, "policies": policies}, indent=1), flush=True)
    return 0


def b_deploy_lock() -> int:
    head = Head(torch.load(C.HEAD32, weights_only=False))
    cal_items = C.load_role("calibration")
    zero = (0.0, 0.0)
    policies, curves = {}, {}
    for arm in C.ARMS:
        tabs = C.load_shards(f"deploy_tables_{arm}", 1, scratch=True)
        scores = []
        for item, t in zip(cal_items, tabs):
            st = doc_structure(item)
            phi = torch.tensor([l["phi"] for l in st], dtype=torch.float64)
            ell = torch.stack([-head.logp(t["reps"][:, i], phi[i].expand(K, -1)) for i in range(len(st))])
            scores.append({"ell": ell, "m": [l["b"] - l["a"] for l in st], "f": torch.stack([l["f"] for l in st])})
        pred = P.predicted_cost(scores, B_GRID, zero)          # argmin(-log p + b r) == argmax(log p - b r)
        policies[arm] = P.choose_parameters(B_GRID, pred, B_TARGETS)
        curves[arm] = [[round(float(B_GRID[j]), 3), float(pred[j])] for j in range(0, len(B_GRID), 20)]
    write_json(C.RESULTS / "policies_B_deploy.json", C.v33_receipt(
        "v33_B_deploy_lock", seed=17, head_sha256=C.sha256_file(C.HEAD32), b_grid={"from": -20, "to": 20, "step": 0.05},
        targets=B_TARGETS, policies=policies, predicted_curve_every_1=curves,
        checkpoint_sha256={arm: C.sha256_file(C.ART / "b" / f"{arm}_s17.pt") for arm in C.ARMS},
        note="locked before any GSM8K-test online execution of experiment B deployment"))
    print(json.dumps(policies, indent=1), flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("step", choices=("A_fit", "B_deploy_lock"))
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    return a_fit(args.nshards) if args.step == "A_fit" else b_deploy_lock()


if __name__ == "__main__":
    raise SystemExit(main())
