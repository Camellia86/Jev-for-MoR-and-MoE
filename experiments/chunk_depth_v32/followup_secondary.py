"""Secondary statistic of the v3.2 follow-up: strict common-cost difference g_B at B = 0.5.

Definition fixed in docs/EXPERIMENT_CONFIG.v3.2-followup.secondary.json before the screen was read.
Per arm, four head deployment parameters: the calibration bracket (from results/chunk-depth-v32-followup/
calibration.json) and two outer parameters chosen on calibration predicted costs (nearest level >= the
high-cost endpoint's predicted cost + 0.01, nearest level <= the low-cost endpoint's predicted cost - 0.01).
The inner pair is taken from the saved screen runs of evaluate.py; the outer pair is executed here.
On the screen, each arm's NLL at exactly B is interpolated between the adjacent pair of policies whose
screen costs bracket B; every bootstrap resample recomputes costs, the pair, the weight and the NLL.
Writes results/chunk-depth-v32-followup/secondary_strict_cost.json and
artifacts/chunk-depth-v32-followup/secondary_strict_cost_per_document.pt.
"""

from __future__ import annotations

import json
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import ROOT, receipt, sha256_file, write_json
from experiments.chunk_depth_v31 import screen as S
from experiments.chunk_depth_v32.common import load_role
from experiments.chunk_depth_v32.stage1 import Head, head_policy

FU_OUT = ROOT / "artifacts" / "chunk-depth-v32-followup"
FU_RESULTS = ROOT / "results" / "chunk-depth-v32-followup"
SECONDARY = ROOT / "docs" / "EXPERIMENT_CONFIG.v3.2-followup.secondary.json"
HEAD = ROOT / "artifacts" / "chunk-depth-v32" / "head.pt"
ARMS = ("E-G", "E-C")
BUDGET, MARGIN, REPEATS, SEED = 0.5, 0.01, 10000, 20261013


def outer_parameters(cal_arm):
    levels = cal_arm["predicted_levels_near_budget"]
    pred = {lvl["param"]: lvl["predicted_cost"] for lvl in levels}
    b_hi, b_lo = cal_arm["bracket"]["param_high_cost"], cal_arm["bracket"]["param_low_cost"]
    above = [lvl for lvl in levels if lvl["predicted_cost"] >= pred[b_hi] + MARGIN]
    below = [lvl for lvl in levels if lvl["predicted_cost"] <= pred[b_lo] - MARGIN]
    if not above or not below:
        raise RuntimeError("outer parameters are not available within the recorded predicted levels")
    b_pp = min(above, key=lambda lvl: lvl["predicted_cost"])
    b_mm = max(below, key=lambda lvl: lvl["predicted_cost"])
    return {"inner_high_cost": b_hi, "inner_low_cost": b_lo, "outer_high_cost": b_pp["param"],
            "outer_low_cost": b_mm["param"], "predicted": {"inner_high_cost": pred[b_hi], "inner_low_cost": pred[b_lo],
                                                           "outer_high_cost": b_pp["predicted_cost"],
                                                           "outer_low_cost": b_mm["predicted_cost"]}}


def interpolate(cost, nll):
    """cost, nll: [R, P] (resamples x policies). Returns L(B) [R] and a success mask [R]."""
    order = cost.argsort(-1)
    c = cost.gather(-1, order)
    l = nll.gather(-1, order)
    out = torch.full(c.shape[:1], float("nan"), dtype=torch.float64)
    ok = torch.zeros(c.shape[:1], dtype=torch.bool)
    for j in range(c.shape[1] - 1):
        c0, c1, l0, l1 = c[:, j], c[:, j + 1], l[:, j], l[:, j + 1]
        hit = (~ok) & (c0 <= BUDGET) & (BUDGET <= c1) & (c1 > c0)
        w = (BUDGET - c0) / (c1 - c0)
        out[hit] = (l0 + w * (l1 - l0))[hit]
        ok |= hit
    return out, ok


def arm_matrices(stats, idx):
    """stats: list over policies of per-document dicts; idx: [R, M] resample indices."""
    nll = torch.tensor([[s["mean_nll"] for s in pol] for pol in stats], dtype=torch.float64)      # [P, M]
    cost = torch.tensor([[s["cost"] for s in pol] for pol in stats], dtype=torch.float64)
    base = torch.tensor([s["base16"] for s in stats[0]], dtype=torch.float64)                   # [M]
    rel = cost[:, idx].sum(-1) / base[idx].sum(-1)                                               # [P, R]
    mean_nll = nll[:, idx].mean(-1)
    return rel.T, mean_nll.T


def main() -> int:
    t0 = time.perf_counter()
    torch.backends.cuda.matmul.allow_tf32 = False
    cal_path, eval_path = FU_RESULTS / "calibration.json", FU_RESULTS / "evaluation.json"
    per_doc_path = FU_OUT / "evaluation_per_document.pt"
    cal = json.loads(cal_path.read_text(encoding="utf-8"))
    evaluation = json.loads(eval_path.read_text(encoding="utf-8"))
    per_doc = torch.load(per_doc_path, map_location="cpu", weights_only=False)
    if per_doc["calibration_sha256"] != sha256_file(cal_path):
        raise RuntimeError("per-document runs do not belong to this calibration")
    screen = load_role("stage2_screen")
    ids = [(it["split"], int(it["row"])) for it in screen]
    head = Head(torch.load(HEAD, map_location="cpu", weights_only=False))
    params = {arm: outer_parameters(cal["calibration"][arm]) for arm in ARMS}
    inner = {}
    for arm in ARMS:
        runs = [r for r in per_doc["runs"] if r["label"] == f"stage2 {arm} head"]
        if len(runs) != 2 or any([(x["split"], x["row"]) for x in r["rows"]] != ids for r in runs):
            raise RuntimeError(f"{arm}: saved screen runs do not match the screen documents")
        inner[arm] = {"high": runs[0]["stats"], "low": runs[1]["stats"]}
        mixed = [cal["calibration"][arm]["bracket"]["weight_high"] * a["mean_nll"]
                 + (1 - cal["calibration"][arm]["bracket"]["weight_high"]) * b["mean_nll"]
                 for a, b in zip(inner[arm]["high"], inner[arm]["low"])]
        if abs(sum(mixed) / len(mixed) - evaluation["policies"][arm]["doc_mean_nll"]) > 1e-12:
            raise RuntimeError(f"{arm}: saved screen runs do not reproduce evaluation.json")
    model = H.load_model()
    lora = S.Lora(model)
    outer, checkpoints = {}, {}
    for arm in ARMS:
        path = FU_OUT / f"pilot_{arm}_s17.pt"
        checkpoints[arm] = sha256_file(path)
        if checkpoints[arm] != cal["checkpoint_sha256"][arm]:
            raise RuntimeError(f"{arm}: checkpoint differs from the calibrated one")
        lora.load(path)
        outer[arm] = {k: S.run_all(model, screen, head_policy(head, params[arm][f"outer_{k}"])) for k in ("high_cost", "low_cost")}
        print(arm, {k: S.summary(v)["relative_cost"] for k, v in outer[arm].items()}, flush=True)
    policies = {arm: [inner[arm]["high"], inner[arm]["low"], outer[arm]["high_cost"], outer[arm]["low_cost"]] for arm in ARMS}
    labels = ["inner_high_cost", "inner_low_cost", "outer_high_cost", "outer_low_cost"]
    M = len(screen)
    full = torch.arange(M)[None]
    g = torch.Generator().manual_seed(SEED)
    idx = torch.randint(0, M, (REPEATS, M), generator=g)
    res = {"policies": {}, "full_sample": {}}
    L_full, L_boot, ok_boot = {}, {}, {}
    for arm in ARMS:
        c_full, l_full = arm_matrices(policies[arm], full)
        res["policies"][arm] = [{"label": lab, "param": params[arm][lab], "screen_cost": float(c), "screen_nll": float(l)}
                                for lab, c, l in zip(labels, c_full[0], l_full[0])]
        L, ok = interpolate(c_full, l_full)
        L_full[arm] = (float(L[0]), bool(ok[0]))
        c_b, l_b = arm_matrices(policies[arm], idx)
        L_boot[arm], ok_boot[arm] = interpolate(c_b, l_b)
    estimable = L_full["E-G"][1] and L_full["E-C"][1]
    both_ok = ok_boot["E-G"] & ok_boot["E-C"]
    fail_share = float((~both_ok).double().mean())
    out = {"definition": str(SECONDARY.relative_to(ROOT)).replace("\\", "/"), "budget": BUDGET, "documents": M,
           "parameters": params, "screen_points": res["policies"],
           "L_at_budget_full_sample": {arm: {"value": v, "bracketed": b} for arm, (v, b) in L_full.items()},
           "bootstrap_failure_share": fail_share}
    if estimable:
        g_full = L_full["E-G"][0] - L_full["E-C"][0]
        out["g_B_EG_minus_EC"] = {"mean": g_full}
        if fail_share == 0.0:
            diff = L_boot["E-G"] - L_boot["E-C"]
            out["g_B_EG_minus_EC"].update({
                "ci95": [float(torch.quantile(diff, 0.025)), float(torch.quantile(diff, 0.975))],
                "lower_one_sided_90": float(torch.quantile(diff, 0.10)), "repeats": REPEATS, "seed": SEED})
        else:
            out["g_B_EG_minus_EC"]["interval"] = "not estimable: some resamples have no bracketing pair"
    else:
        out["g_B_EG_minus_EC"] = "not estimable: an arm has no bracketing pair on the full screen sample"
    torch.save({"outer": outer, "params": params}, FU_OUT / "secondary_strict_cost_per_document.pt")
    out["seconds"] = time.perf_counter() - t0
    write_json(FU_RESULTS / "secondary_strict_cost.json", receipt(
        "chunk_depth_v3_2_followup_secondary_strict_cost", protocol_followup="jev-chunk-depth-v3.2-followup",
        provenance={"secondary_sha256": sha256_file(SECONDARY), "script_sha256": sha256_file(ROOT / "experiments" /
                    "chunk_depth_v32" / "followup_secondary.py"), "calibration_sha256": sha256_file(cal_path),
                    "evaluation_sha256": sha256_file(eval_path), "per_document_sha256": sha256_file(per_doc_path),
                    "checkpoint_sha256": checkpoints, "head_sha256": sha256_file(HEAD)}, **out))
    print(json.dumps({k: out[k] for k in ("parameters", "screen_points", "L_at_budget_full_sample",
                                          "bootstrap_failure_share", "g_B_EG_minus_EC")}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
