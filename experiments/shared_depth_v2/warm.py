"""Protocol sections 5.7 and 7.3: zero-LoRA control representations, readout warm-up per lambda,
and the lambda rule (warm-set uncalibrated expected plan cost closest to 1 in log space).

Outputs: artifacts/.../pilot/control_rep_zero_<role>.pt, readouts.pt; receipt results/.../warm.json
"""

from __future__ import annotations

import argparse
import json
import math

import torch

from . import cost, policy
from . import huginn as H
from .common import ARTIFACTS, RESULTS, Timer, load_config, receipt, write_json
from .extract_features import load_features

WARM_SEED = 20261011


def control_reps(model, role: str) -> torch.Tensor:
    header = json.loads((ARTIFACTS / "features" / f"{role}.json").read_text(encoding="utf-8"))
    fcp = H.freqs(model, torch.arange(128))
    out = torch.empty(len(header["documents"]), 5280, dtype=torch.float32)
    with torch.no_grad():
        for j in range(len(header["documents"])):
            prefix = load_features(role, j, header)["prefix"].cuda()[None]
            out[j] = H.coda(model, prefix, fcp)[0, 127].float().cpu()
    return out


def base_mean_nll(role: str) -> torch.Tensor:
    return torch.load(ARTIFACTS / "features" / f"{role}.base_nll.pt")["base_nll"].mean(-1).double()


def new_readout(seed: int = WARM_SEED) -> torch.nn.Linear:
    g = torch.Generator(device="cpu").manual_seed(seed)
    lin = torch.nn.Linear(5280, 4)
    bound = 1 / math.sqrt(5280)
    with torch.no_grad():
        lin.weight.copy_(torch.empty(4, 5280).uniform_(-bound, bound, generator=g))
        lin.bias.zero_()
    return lin


def train_readout(z: torch.Tensor, nll: torch.Tensor, f: torch.Tensor, lam: float, cfg: dict) -> tuple[torch.nn.Linear, list]:
    r = cfg["readout"]
    lin = new_readout().cuda()
    opt = torch.optim.AdamW(lin.parameters(), lr=r["warm_lr"], weight_decay=r["warm_weight_decay"])
    g = torch.Generator(device="cpu").manual_seed(WARM_SEED)
    z, c = z.cuda(), (nll + lam * f[None]).cuda()
    uniform = torch.full((4,), 0.25, dtype=torch.float64, device="cuda")
    n, bs = z.shape[0], r["warm_batch"]
    order, pos, log = torch.randperm(n, generator=g), 0, []
    for step in range(r["warm_updates"]):
        if pos + bs > n:
            order, pos = torch.randperm(n, generator=g), 0
        idx = order[pos:pos + bs].cuda()
        pos += bs
        mu = policy.mean_policy(lin(z[idx]))
        loss = (policy.route_exact(mu, c[idx]) + policy.auxiliary(mu, c[idx], uniform.expand_as(mu))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 50 == 0 or step == r["warm_updates"] - 1:
            log.append({"step": step, "loss": float(loss)})
    return lin.cpu(), log


def expected(mu: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    return (mu * values).sum(-1)


def main() -> int:
    argparse.ArgumentParser(description=__doc__).parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = load_config()
    costs: dict = {}
    out_dir = ARTIFACTS / "pilot"
    out_dir.mkdir(parents=True, exist_ok=True)
    with Timer(costs, "model_load_seconds"):
        model = H.load_model()
    reps = {}
    with Timer(costs, "control_rep_seconds"):
        for role in ("warm", "pilot_train", "pilot_cal", "pilot_screen"):
            reps[role] = control_reps(model, role)
            torch.save(reps[role], out_dir / f"control_rep_zero_{role}.pt")
    norm = cost.normalized()
    f_plan = torch.tensor([norm["f_plan"][d] for d in H.DEPTHS], dtype=torch.float64)
    f_static = torch.tensor([norm["f_static"][d] for d in H.DEPTHS], dtype=torch.float64)
    nll = {role: base_mean_nll(role) for role in reps}

    # Static frozen comparator on pilot_cal for the diagnostic NLL at budget 1.
    cal_points = [{"name": f"frozen@{d}", "cost": float(f_static[k]), "nll": float(nll["pilot_cal"][:, k].mean())}
                  for k, d in enumerate(H.DEPTHS)]
    static_cal = policy.best_static_mixture(cal_points, 1.0)

    rows, readouts = [], {}
    with Timer(costs, "readout_training_seconds"):
        for lam in cfg["objective"]["lambda_grid"]:
            lin, log = train_readout(reps["warm"], nll["warm"], f_plan, lam, cfg)
            readouts[lam] = lin.state_dict()
            with torch.no_grad():
                mu = {role: policy.mean_policy(lin(reps[role])) for role in reps}
            cbar = float(expected(mu["warm"], f_plan[None]).mean())
            cal = policy.calibrate(mu["pilot_cal"], f_plan, 1.0)
            screen_mu_b = policy.tilt(mu["pilot_screen"], f_plan, cal["b"])
            rows.append({
                "lambda": lam, "warm_uncalibrated_mean_plan_cost": cbar, "abs_log_cost": abs(math.log(cbar)),
                "warm_expected_nll_b0": float(expected(mu["warm"], nll["warm"]).mean()),
                "pilot_cal_expected_nll_b0": float(expected(mu["pilot_cal"], nll["pilot_cal"]).mean()),
                "warm_minus_cal_expected_cost_gap": float(
                    (expected(mu["warm"], nll["warm"] + lam * f_plan[None]).mean()
                     - expected(mu["pilot_cal"], nll["pilot_cal"] + lam * f_plan[None]).mean())),
                "diag_pilot_screen_nll_at_budget1": float(expected(screen_mu_b, nll["pilot_screen"]).mean()),
                "diag_pilot_screen_cost_at_budget1": float(expected(screen_mu_b, f_plan[None]).mean()),
                "diag_calibration": cal,
                "warm_mean_policy": [float(v) for v in mu["warm"].mean(0)],
                "train_log": log,
            })
    best = min(rows, key=lambda r: (round(r["abs_log_cost"], 12), r["lambda"]))
    selected = best["lambda"]
    torch.save({"readouts": readouts, "selected_lambda": selected, "init": "uniform(+-1/sqrt(5280)) weight, zero bias, seed 20261011"},
               out_dir / "readouts.pt")
    payload = receipt("pilot_warm_readout_and_lambda", protocol_revision="2.1", selected_lambda=selected,
                      rule=cfg["objective"]["lambda_rule"], f_plan=norm["f_plan"], f_static=norm["f_static"],
                      static_frozen_pilot_cal=static_cal,
                      base_mean_nll={role: [float(v) for v in nll[role].mean(0)] for role in nll},
                      rows=rows, costs=costs)
    write_json(RESULTS / "warm.json", payload)
    print(json.dumps({"selected_lambda": selected,
                      "rows": [{k: r[k] for k in ("lambda", "warm_uncalibrated_mean_plan_cost", "warm_expected_nll_b0",
                                                   "diag_pilot_screen_nll_at_budget1", "warm_mean_policy")} for r in rows],
                      "static": static_cal, "costs": costs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
