"""Statistics of v3.3 (plan sections 4.5-4.8, 5.2-5.3, 6). CPU only.

    python -m experiments.chunk_depth_v33.analyze A --nshards n
    python -m experiments.chunk_depth_v33.analyze B --nshards n
"""

from __future__ import annotations

import argparse
import json
import math
import statistics

import torch

from experiments.shared_depth_v2.common import write_json
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v33 import common as C

T_975_DF2 = 4.302652729911275


def boot_index(M: int, repeats: int = None) -> torch.Tensor:
    g = torch.Generator().manual_seed(C.BOOT_SEED)
    return torch.randint(0, M, (repeats or C.REPEATS, M), generator=g)


def summarize(boot: torch.Tensor, point: float) -> dict:
    q = lambda p: float(torch.quantile(boot, p))
    return {"mean": point, "ci95": [q(0.025), q(0.975)], "lower_one_sided_95": q(0.05),
            "lower_one_sided_90": q(0.10), "p_one_sided_le_0": float((boot <= 0).double().mean()),
            "bootstrap_sd": float(boot.std())}


def family_matrices(policy_stats: list[list[dict]], idx: torch.Tensor, chunk: int = 500):
    """policy_stats[p][doc] -> (relative cost, mean NLL), each [resamples, P]; resamples processed in chunks."""
    nll = torch.tensor([[s["mean_nll"] for s in pol] for pol in policy_stats], dtype=torch.float64)
    cost = torch.tensor([[s["cost"] for s in pol] for pol in policy_stats], dtype=torch.float64)
    base = torch.tensor([s["base16"] for s in policy_stats[0]], dtype=torch.float64)
    rels, nlls = [], []
    for s in range(0, len(idx), chunk):
        ii = idx[s:s + chunk]
        rels.append((cost[:, ii].sum(-1) / base[ii].sum(-1)).T)
        nlls.append(nll[:, ii].mean(-1).T)
    return torch.cat(rels), torch.cat(nlls)


def interpolate(cost: torch.Tensor, nll: torch.Tensor, budget: float):
    """Loss of the cost-B two-point mixture of the adjacent policies (in cost order) that bracket B."""
    order = cost.argsort(-1)
    c, l = cost.gather(-1, order), nll.gather(-1, order)
    out = torch.full(c.shape[:1], float("nan"), dtype=torch.float64)
    ok = torch.zeros(c.shape[:1], dtype=torch.bool)
    for j in range(c.shape[1] - 1):
        c0, c1, l0, l1 = c[:, j], c[:, j + 1], l[:, j], l[:, j + 1]
        hit = (~ok) & (c0 <= budget) & (budget <= c1) & (c1 > c0)
        out[hit] = (l0 + (budget - c0) / (c1 - c0) * (l1 - l0))[hit]
        ok |= hit
    return out, ok


def strict_difference(fam_a, fam_b, budget, idx_full, idx_boot):
    """L_a(B) - L_b(B) with every resample recomputing costs, bracketing pairs and weights."""
    La, oka = interpolate(*family_matrices(fam_a, idx_full), budget)
    Lb, okb = interpolate(*family_matrices(fam_b, idx_full), budget)
    out = {"L_first": float(La[0]), "L_second": float(Lb[0]), "bracketed_full_sample": bool(oka[0] and okb[0])}
    if not out["bracketed_full_sample"]:
        out["status"] = "not estimable: no bracketing pair on the full sample"
        return out
    ba, okA = interpolate(*family_matrices(fam_a, idx_boot), budget)
    bb, okB = interpolate(*family_matrices(fam_b, idx_boot), budget)
    fail = float((~(okA & okB)).double().mean())
    out["bootstrap_failure_share"] = fail
    point = float(La[0] - Lb[0])
    if fail > 0:
        out.update({"mean": point, "status": "interval not estimable: some resamples have no bracketing pair"})
    else:
        out.update(summarize(ba - bb, point))
        out["status"] = "ok"
    return out


def conclusion(res: dict) -> str:
    if res.get("status") != "ok":
        return "not_estimable"
    lo, hi, lb = res["ci95"][0], res["ci95"][1], res["lower_one_sided_95"]
    if lb > 0:
        if lo >= 0.005:
            return "gain_detected_reaches_practical_threshold"
        if hi < 0.005:
            return "gain_detected_below_practical_threshold"
        return "gain_detected_threshold_undetermined"
    if hi < 0.005:
        return "no_gain_detected_below_practical_threshold"
    return "no_gain_detected_threshold_undetermined"


def depth_shares(stats: list[dict]) -> dict:
    counts = {d: 0 for d in R.V3}
    for s in stats:
        for d in s["depths"]:
            counts[d] += 1
    n = max(1, sum(counts.values()))
    return {str(d): c / n for d, c in counts.items()}


# ----------------------------------------------------------------------------- experiment A

def analyze_a(nshards: int) -> dict:
    lock = json.loads((C.RESULTS / "policies_A.json").read_text(encoding="utf-8"))
    rows = C.load_shards("online_A", nshards)
    names = list(rows[0]["policies"])
    if any(set(r["policies"]) != set(names) for r in rows):
        raise RuntimeError("documents were executed with different policy sets")
    fams = {f: [n for n in names if n.startswith(f + "_")] for f in ("R", "H", "U")}
    stats = {n: [r["policies"][n] for r in rows] for n in names}
    M = len(rows)
    full = torch.arange(M)[None]
    boot = boot_index(M)
    out = {"documents": M, "parts": {p: sum(r["part"] == p for r in rows) for p in ("train_50_64", "test")},
           "policies": {n: {"relative_cost": sum(s["cost"] for s in stats[n]) / sum(s["base16"] for s in stats[n]),
                            "doc_mean_nll": sum(s["mean_nll"] for s in stats[n]) / M,
                            "depth_share": depth_shares(stats[n])} for n in names}}
    fam = {f: [stats[n] for n in fams[f]] for f in fams}
    res = {}
    for b in C.BUDGETS:
        key = str(b)
        res[key] = {"g_A_R_minus_H": strict_difference(fam["R"], fam["H"], b, full, boot),
                    "U_minus_R": strict_difference(fam["U"], fam["R"], b, full, boot),
                    "U_minus_H": strict_difference(fam["U"], fam["H"], b, full, boot)}
        res[key]["g_A_R_minus_H"]["conclusion"] = conclusion(res[key]["g_A_R_minus_H"])
        for part in ("train_50_64", "test"):
            sel = [i for i, r in enumerate(rows) if r["part"] == part]
            sub = {f: [[pol[i] for i in sel] for pol in fam[f]] for f in ("R", "H")}
            res[key][f"g_A_{part}"] = strict_difference(sub["R"], sub["H"], b, torch.arange(len(sel))[None],
                                                        boot_index(len(sel)))
    out["strict_common_cost"] = res
    out["primary"] = {"budget": 0.5, **res["0.5"]["g_A_R_minus_H"]}
    agree = {}
    for n in fams["H"]:
        a = [sum(r["agree"][n][0] for r in rows), sum(r["agree"][n][1] for r in rows)]
        agree[n] = {"agree": a[0], "decisions": a[1], "rate": a[0] / max(1, a[1])}
    out["H_table_vs_online_agreement"] = agree
    if C.shards_complete("calcheck", nshards):
        cal = C.load_shards("calcheck", nshards)
        pred = {f"{fam_}_B{b}_t{e['target']:.3f}": e["predicted_cost"]
                for fam_ in ("R", "H") for b in C.BUDGETS for e in lock["policies"][fam_][str(b)]}
        chk = {}
        for n in cal[0]:
            online = sum(r[n]["cost"] for r in cal) / sum(r[n]["base16"] for r in cal)
            chk[n] = {"online": online, "predicted": pred[n], "abs_diff": abs(online - pred[n]),
                      "passed": abs(online - pred[n]) <= 0.01}
        out["calibration_check"] = chk
    write_json(C.RESULTS / "analysis_A.json", C.v33_receipt("v33_analysis_A", **out))
    p = out["primary"]
    print(json.dumps({k: p.get(k) for k in ("mean", "ci95", "lower_one_sided_95", "status", "conclusion")}, indent=1), flush=True)
    return out


# ----------------------------------------------------------------------------- experiment B

def holm(pvals: dict, alpha: float = 0.05) -> dict:
    order = sorted(pvals, key=lambda k: pvals[k])
    out, still = {}, True
    for rank, k in enumerate(order):
        thr = alpha / (len(order) - rank)
        rej = still and pvals[k] <= thr
        out[k] = {"p": pvals[k], "threshold": thr, "rejected": rej}
        still = rej
    return out


def analyze_b(nshards: int) -> dict:
    rows = C.load_shards("eval_B", nshards)
    M = len(rows)
    names = list(rows[0]["models"])
    for r in rows:
        d0 = r["models"][names[0]]["depths"]
        if any(r["models"][n]["depths"] != d0 for n in names) or len({r["models"][n]["cost"] for n in names}) != 1:
            raise RuntimeError(f"fixed schedule differs across models for {r['split']}:{r['row']}")
    L = {n: torch.tensor([r["models"][n]["mean_nll"] for r in rows], dtype=torch.float64) for n in names}
    boot = boot_index(M)
    per_seed, pooled = {}, {}
    for s in C.SEEDS:
        hC = L[f"C_s{s}"] - L[f"G_s{s}"]
        hD = L[f"D_s{s}"] - L[f"G_s{s}"]
        per_seed[str(s)] = {"h_C": summarize(hC[boot].mean(1), float(hC.mean())),
                            "h_D": summarize(hD[boot].mean(1), float(hD.mean())),
                            "Delta_CD": summarize((hC - hD)[boot].mean(1), float((hC - hD).mean()))}
    qty = {"h_C": lambda s: L[f"C_s{s}"] - L[f"G_s{s}"], "h_D": lambda s: L[f"D_s{s}"] - L[f"G_s{s}"],
           "Delta_CD": lambda s: L[f"C_s{s}"] - L[f"D_s{s}"]}
    for k, f in qty.items():
        x = torch.stack([f(s) for s in C.SEEDS]).mean(0)
        pooled[k] = summarize(x[boot].mean(1), float(x.mean()))
        seeds = [float(f(s).mean()) for s in C.SEEDS]
        m, sd = statistics.mean(seeds), statistics.stdev(seeds) if len(seeds) > 1 else 0.0
        pooled[k]["per_seed_means"] = seeds
        pooled[k]["seed_t_interval_df2"] = [m - T_975_DF2 * sd / math.sqrt(len(seeds)), m + T_975_DF2 * sd / math.sqrt(len(seeds))]
        pooled[k]["all_seeds_same_sign_as_pooled"] = all((v > 0) == (pooled[k]["mean"] > 0) for v in seeds)
    tests = holm({"B1_h_C": pooled["h_C"]["p_one_sided_le_0"], "B2_Delta_CD": pooled["Delta_CD"]["p_one_sided_le_0"]})
    out = {"documents": M, "fixed_schedule": {"artifact": "artifacts/chunk-depth-v32/R.pt", "nu": C.FIXED_NU},
           "model_mean_nll": {n: float(v.mean()) for n, v in L.items()}, "per_seed": per_seed, "pooled": pooled,
           "holm": tests, "depth_share": depth_shares([r["models"][names[0]] for r in rows])}
    train = {}
    for arm, s in [(a, s) for s in C.SEEDS for a in C.ARMS]:
        p = C.RESULTS / "b" / f"train_{arm}_s{s}.json"
        if p.exists():
            lg = json.loads(p.read_text(encoding="utf-8"))["log"]
            tail = lg[-10:]
            train[f"{arm}_s{s}"] = {"depth8_nll_last10": statistics.mean(e["depth8_nll"] for e in tail),
                                   "depth8_nll_first10": statistics.mean(e["depth8_nll"] for e in lg[:10]),
                                   "mean_delta_norm": statistics.mean(e["parameter_delta_norm"] for e in lg)}
    out["training"] = train
    if C.shards_complete("deploy_online_B", nshards):
        drows = C.load_shards("deploy_online_B", nshards)
        fam = {arm: [[r["policies"][n] for r in drows] for n in drows[0]["policies"] if n.startswith(arm + "_")]
               for arm in C.ARMS}
        Md = len(drows)
        full, bt = torch.arange(Md)[None], boot_index(Md)
        out["deployment_seed17_test"] = {"documents": Md,
                                         "G_minus_C": strict_difference(fam["G"], fam["C"], 0.5, full, bt),
                                         "G_minus_D": strict_difference(fam["G"], fam["D"], 0.5, full, bt)}
    write_json(C.RESULTS / "analysis_B.json", C.v33_receipt("v33_analysis_B", **out))
    print(json.dumps({"pooled": {k: {kk: v[kk] for kk in ("mean", "ci95", "p_one_sided_le_0")} for k, v in pooled.items()},
                      "holm": tests}, indent=1), flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("part", choices=("A", "B"))
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    analyze_a(args.nshards) if args.part == "A" else analyze_b(args.nshards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
