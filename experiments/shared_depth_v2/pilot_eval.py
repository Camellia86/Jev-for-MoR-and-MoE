"""Protocol section 7.4: pilot go/no-go (predictability probe) and diagnostics on pilot_screen.

Static comparator: frozen four-depth mixture at cost <= 1 chosen on pilot_cal (f_static).
Probe: warm readout + zero LoRA, calibrated on pilot_cal to the comparator's cost (f_plan),
evaluated on all 384 pilot_screen documents. Gate: mean improvement >= 0.002 nat and one-sided
95% document-bootstrap lower bound > 0.
"""

from __future__ import annotations

import argparse
import json

import torch

from . import cost, policy
from . import huginn as H
from .common import ARTIFACTS, RESULTS, Timer, load_config, receipt, write_json
from .extract_features import load_features
from .train import ARMS, load_readout

ROLES = ("pilot_cal", "pilot_screen")


def arm_outputs(model, readout, role: str):
    """Per-document mean NLL at 4 depths [N,4] and control policy mu [N,4] for current LoRA state."""
    header = json.loads((ARTIFACTS / "features" / f"{role}.json").read_text(encoding="utf-8"))
    win = torch.load(ARTIFACTS / "data" / "windows.pt")
    fc, fcp = H.freqs(model, torch.arange(255)), H.freqs(model, torch.arange(128))
    n = len(header["documents"])
    nll, mu = torch.empty(n, 4, dtype=torch.float64), torch.empty(n, 4, dtype=torch.float64)
    with torch.no_grad():
        for j in range(n):
            feats = load_features(role, j, header)
            targets = win[header["manifest_indices"][j], 128:256].long().cuda()
            for k, d in enumerate(H.DEPTHS):
                logits = H.head(model, H.coda(model, feats[d].cuda()[None], fc))[0, 127:]
                nll[j, k] = float(torch.nn.functional.cross_entropy(logits, targets))
            z = H.coda(model, feats["prefix"].cuda()[None], fcp)[0, 127].float()
            mu[j] = policy.mean_policy(readout(z)).cpu()
    return nll, mu


def static_points(nll_cal: torch.Tensor, f_static: torch.Tensor, prefix: str):
    return [{"name": f"{prefix}@{d}", "cost": float(f_static[k]), "nll": float(nll_cal[:, k].mean())}
            for k, d in enumerate(H.DEPTHS)]


def mixture_weights(mix: dict) -> dict:
    return {name: w for name, w in mix["support"]}


def apply_mixture(mix: dict, tables: dict) -> torch.Tensor:
    """tables: name -> per-doc NLL vector. Returns per-doc expected NLL of the mixture."""
    out = None
    for name, w in mix["support"]:
        v = w * tables[name]
        out = v if out is None else out + v
    return out


def oracle(nll_select: torch.Tensor, nll_eval: torch.Tensor, f: torch.Tensor, budget: float) -> dict:
    """Lagrangian per-document depth choice with boundary randomisation (section 7.4).

    Chooses argmin nll_select + nu*f per document. Bisection finds the smallest nu whose mean
    cost is within budget; the policy then mixes that choice with the choice just below it so
    the mean cost equals the budget exactly. Returns expected cost and expected nll_eval.
    """
    def pick(nu: float):
        choice = (nll_select + nu * f[None]).argmin(1)
        return choice, float(f[choice].mean()), float(nll_eval.gather(1, choice[:, None]).mean())

    choice0, c0, q0 = pick(0.0)
    if c0 <= budget:
        return {"nu": 0.0, "cost": c0, "nll": q0, "choice": choice0, "mix_weight_low_nu": 0.0}
    lo, hi = 0.0, 1.0
    while pick(hi)[1] > budget:
        hi *= 2
        if hi > 1e6:
            raise ValueError("budget below minimum cost")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if pick(mid)[1] > budget:
            lo = mid
        else:
            hi = mid
    ch_hi, c_hi, q_hi = pick(hi)
    ch_lo, c_lo, q_lo = pick(lo)
    w = 0.0 if c_lo == c_hi else (budget - c_hi) / (c_lo - c_hi)
    w = min(max(w, 0.0), 1.0)
    return {"nu": hi, "cost": w * c_lo + (1 - w) * c_hi, "nll": w * q_lo + (1 - w) * q_hi,
            "choice": ch_hi, "mix_weight_low_nu": w}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tag", default="pilot")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = load_config()
    costs: dict = {}
    norm = cost.normalized()
    f_plan = torch.tensor([norm["f_plan"][d] for d in H.DEPTHS], dtype=torch.float64)
    f_static = torch.tensor([norm["f_static"][d] for d in H.DEPTHS], dtype=torch.float64)
    base = {r: torch.load(ARTIFACTS / "features" / f"{r}.base_nll.pt")["base_nll"].double() for r in ROLES}
    frozen = {r: base[r].mean(-1) for r in ROLES}

    static_mix = policy.best_static_mixture(static_points(frozen["pilot_cal"], f_static, "frozen"), 1.0)
    c_static = static_mix["cost"]
    frozen_tables = {f"frozen@{d}": frozen["pilot_screen"][:, k] for k, d in enumerate(H.DEPTHS)}
    static_screen = apply_mixture(static_mix, frozen_tables)

    with Timer(costs, "model_load_seconds"):
        model = H.load_model()
    readout, lam = load_readout()
    z0 = {r: torch.load(ARTIFACTS / "pilot" / f"control_rep_zero_{r}.pt").cuda() for r in ROLES}
    with torch.no_grad():
        mu0 = {r: policy.mean_policy(readout(z0[r])).cpu() for r in ROLES}
    cal = policy.calibrate(mu0["pilot_cal"], f_plan, c_static)
    mu_b = policy.tilt(mu0["pilot_screen"], f_plan, cal["b"])
    probe_screen = (mu_b * frozen["pilot_screen"]).sum(-1)
    probe_gain = policy.bootstrap_mean(static_screen - probe_screen)
    probe_gate = probe_gain["mean"] >= cfg["pilot_go"]["predictability_probe"]["min_point_estimate_nats"] \
        and probe_gain["lower_one_sided_95"] > 0
    probe = {"calibration": cal, "screen_expected_cost": float((mu_b * f_plan).sum(-1).mean()),
             "screen_expected_nll": float(probe_screen.mean()), "gain_vs_static": probe_gain,
             "screen_mean_policy": [float(v) for v in mu_b.mean(0)], "passed": probe_gate}

    hind = oracle(frozen["pilot_screen"], frozen["pilot_screen"], f_plan, c_static)
    first = base["pilot_screen"][:, :, :64].mean(-1)
    second = base["pilot_screen"][:, :, 64:].mean(-1)
    split = oracle(first, second, f_plan, c_static)
    static_second = apply_mixture(static_mix, {f"frozen@{d}": second[:, k] for k, d in enumerate(H.DEPTHS)})
    diagnostics = {
        "hindsight_oracle": {"nu": hind["nu"], "cost": hind["cost"], "nll": hind["nll"],
                             "gain_vs_static": float(static_screen.mean() - hind["nll"]),
                             "depth_share": [float((hind["choice"] == k).double().mean()) for k in range(4)]},
        "split_half_oracle": {"nu": split["nu"], "cost": split["cost"], "nll_second_half": split["nll"],
                              "static_nll_second_half": float(static_second.mean()),
                              "gain_vs_static": float(static_second.mean() - split["nll"])},
    }

    arms = {}
    with Timer(costs, "arm_eval_seconds"):
        modules = H.attach_lora(model, cfg["lora"]["rank"], cfg["lora"]["alpha"], seed=0)
        params = H.lora_parameters(modules)
        for arm in ARMS:
            path = ARTIFACTS / args.tag / f"{arm}_s{args.seed}"
            ckpts = sorted(path.glob("lora_u*.pt"), key=lambda p: int(p.stem.split("_u")[1]))
            if not ckpts:
                continue
            blob = torch.load(ckpts[-1])
            with torch.no_grad():
                for p, v in zip(params, blob["lora"]):
                    p.copy_(v.cuda())
            out = {r: arm_outputs(model, readout, r) for r in ROLES}
            nll_cal, mu_cal = out["pilot_cal"]
            nll_scr, mu_scr = out["pilot_screen"]
            arm_static = policy.best_static_mixture(static_points(nll_cal, f_static, arm), 1.0)
            arm_tables = {f"{arm}@{d}": nll_scr[:, k] for k, d in enumerate(H.DEPTHS)}
            entry = {"checkpoint": str(ckpts[-1]), "update": blob["update"],
                     "screen_nll_by_depth": [float(v) for v in nll_scr.mean(0)],
                     "own_static": {"mixture": arm_static,
                                    "gain_vs_frozen_static": policy.bootstrap_mean(
                                        static_screen - apply_mixture(arm_static, arm_tables))}}
            if arm in ("E", "J"):
                c = policy.calibrate(mu_cal, f_plan, c_static)
                mb = policy.tilt(mu_scr, f_plan, c["b"])
                dyn = (mb * nll_scr).sum(-1)
                entry["dynamic"] = {"calibration": c, "screen_cost": float((mb * f_plan).sum(-1).mean()),
                                    "screen_nll": float(dyn.mean()),
                                    "mean_policy": [float(v) for v in mb.mean(0)],
                                    "gain_vs_frozen_static": policy.bootstrap_mean(static_screen - dyn),
                                    "gain_vs_own_static": policy.bootstrap_mean(
                                        apply_mixture(arm_static, arm_tables) - dyn)}
            arms[arm] = entry
        H.detach_lora(model)

    train_receipts = {arm: json.loads((RESULTS / f"{args.tag}_train_{arm}_s{args.seed}.json").read_text(encoding="utf-8"))
                      for arm in ARMS if (RESULTS / f"{args.tag}_train_{arm}_s{args.seed}.json").exists()}
    sec_per_update = {arm: r["costs"]["train_seconds"] / r["updates"] for arm, r in train_receipts.items()}
    extraction = json.loads((RESULTS / "feature_extraction.json").read_text(encoding="utf-8"))["roles"]
    sec_per_doc = max(v["seconds_per_doc"] for v in extraction.values())
    formal_docs = 4096 + 256 + 512 + 2048
    forecast = {
        "assumption": "RTX 5080 timings used as the A10 estimate for compute-bound phases; decode scaled by "
                      "the bandwidth ratio 960/600; to be replaced by the 100-update A10 measurement",
        "feature_extraction_gpu_hours": (formal_docs + 2 * 128) * sec_per_doc / 3600,
        "training_gpu_hours": (sum(sec_per_update.values()) / max(1, len(sec_per_update))) * 16384 / 3600
        if sec_per_update else None,
        "seconds_per_update_by_arm": sec_per_update,
    }
    if forecast["training_gpu_hours"] is not None:
        forecast["training_gpu_hours_x1.2"] = forecast["training_gpu_hours"] * 1.2
    caps = cfg["resources"]["formal"]["phase_caps"]
    forecast["within_caps"] = (forecast["feature_extraction_gpu_hours"] * 1.2 <= caps["cache_and_runtime"]
                               and (forecast.get("training_gpu_hours_x1.2") or 0) <= caps["training"])

    impl = json.loads((RESULTS / "impl_checks.json").read_text(encoding="utf-8"))
    go = {"technical_checks": impl["status"] == "passed", "resource_forecast": forecast["within_caps"],
          "predictability_probe": probe_gate}
    payload = receipt("pilot_go_no_go", protocol_revision="2.1", decision="go" if all(go.values()) else "no_go",
                      criteria=go, lambda_=lam, static_comparator=static_mix,
                      static_screen_nll=float(static_screen.mean()), probe=probe, diagnostics=diagnostics,
                      arms=arms, forecast=forecast, costs=costs)
    write_json(RESULTS / f"{args.tag}_eval.json", payload)
    print(json.dumps({"decision": payload["decision"], "criteria": go, "static": static_mix,
                      "probe_gain": probe_gain, "diagnostics": diagnostics,
                      "arms": {a: {"dyn": (e.get("dynamic") or {}).get("gain_vs_frozen_static"),
                                   "own_static": e["own_static"]["gain_vs_frozen_static"]} for a, e in arms.items()},
                      "forecast": forecast}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
