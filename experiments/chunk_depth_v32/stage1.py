"""Stage 1 step 3 (plan v3.2 sections 4.5-4.7): calibration, technical checks, gate and diagnostics.

theta_S is loaded throughout. Head and R are calibrated on the calibration set (612 documents) by the
predicted-cost search of v3.1 screen.calibrate (fine grid from calibration table scores, online execution
near the budget, alignment on online results only), then evaluated on stage1_screen (367 documents).
The gate part is written to results/chunk-depth-v32/stage1.json before the diagnostics start.
"""

from __future__ import annotations

import json
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31 import screen as S
from experiments.chunk_depth_v31.train import K, doc_structure, load_kv
from experiments.chunk_depth_v32.common import OUT, RESULTS, THETA_S, check_theta_s, load_role
from experiments.chunk_depth_v32.head import ridge_from_state, soft_labels, build_rows

B_FINE = [round(-20 + i * 0.05, 2) for i in range(801)]


class Head:
    def __init__(self, blob):
        self.W, self.beta, self.m, self.sd = blob["W"], blob["beta"], blob["m"], blob["sd"]

    def logp(self, z, phi):
        """z [..., 5280], phi [..., 10] -> log p [..., K] (float64)."""
        x = torch.cat([(z.double() - self.m) / self.sd, phi.double()], -1)
        return torch.log_softmax(x @ self.W.T + self.beta, -1)


def head_policy(head, b, record=None):
    def pol(z, dz, c, prev_len, hist):
        k = int((head.logp(z, torch.tensor(R.pos_feats(c, prev_len))) - b * R.R_REL).argmax())
        if record is not None:
            record.append(k)
        return R.V3[k]
    return pol


def scores_head(items, tabs, head):
    out = []
    for item, t in zip(items, tabs):
        st = doc_structure(item)
        phi = torch.tensor([l["phi"] for l in st], dtype=torch.float64)
        rows = [head.logp(t["reps"][:, i], phi[i].expand(K, -1)) for i in range(len(st))]
        out.append({"s": rows, "f": [l["f"] for l in st]})
    return out


def scores_rep_ridge(items, tabs, probe):
    out = []
    eye = torch.eye(K, dtype=torch.float64)
    for item, t in zip(items, tabs):
        st = doc_structure(item)
        rows = []
        for i, l in enumerate(st):
            x = torch.cat([t["reps"][:, i].double(), eye, torch.tensor(l["phi"], dtype=torch.float64).expand(K, -1)], 1)
            rows.append(-probe(x))
        out.append({"s": rows, "f": [l["f"] for l in st]})
    return out


def online_vs_predicted(scan, curve):
    pred = {}
    for lvl in curve:
        lo, hi = lvl["param_range"]
        pred[lvl["param"]] = lvl["predicted_cost"]
    return [{"param": p, "online": S.summary(s)["relative_cost"], "predicted": pred.get(p)} for p, s in sorted(scan.items())]


def bracket_levels(curve):
    costs = sorted(lvl["predicted_cost"] for lvl in curve)
    below = [c for c in costs if c <= S.BUDGET]
    above = [c for c in costs if c >= S.BUDGET]
    return (below[-1] if below else None), (above[0] if above else None)


def per_doc(ev):
    return ev["doc"]


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    t_start = time.perf_counter()
    sha = check_theta_s()
    kappa_s, kappa_e = load_kv()
    cal, screen = load_role("calibration"), load_role("stage1_screen")
    cal_tabs = torch.load(OUT / "tables_calibration.pt")
    head_blob = torch.load(OUT / "head.pt")
    head = Head(head_blob)
    ridge = ridge_from_state(torch.load(OUT / "R.pt"))
    probe = ridge_from_state(torch.load(OUT / "rep_ridge.pt"))
    head_json = json.loads((RESULTS / "head.json").read_text(encoding="utf-8"))
    model = H.load_model()
    lora = S.Lora(model)
    lora.load(THETA_S)
    timing = []
    result = {"theta_s_sha256": sha, "budget": S.BUDGET, "lambda": head_blob["lambda"], "gamma": head_blob["gamma"],
              "documents": {"calibration": len(cal), "stage1_screen": len(screen)}}

    # calibration on the calibration set
    h_scan, h_br, h_curve = S.calibrate(model, cal, lambda b: head_policy(head, b), scores_head(cal, cal_tabs, head),
                                        B_FINE, "head", timing)
    r_scan, r_br, r_curve = S.calibrate(model, cal, lambda nu: S.position_policy(ridge, nu), S.scores_position(cal, ridge),
                                        S.NU_FINE, "R", timing)
    result["calibration"] = {"head": {**S.scan_record(h_scan, h_br, h_curve),
                                      "online_vs_predicted": online_vs_predicted(h_scan, h_curve)},
                             "R": {**S.scan_record(r_scan, r_br, r_curve),
                                   "online_vs_predicted": online_vs_predicted(r_scan, r_curve)}}

    # technical checks
    lam_row = next(r for r in head_json["lambda_rule"]["rows"] if r["lambda"] == head_blob["lambda"])
    rows = build_rows(cal, cal_tabs, kappa_s, kappa_e)
    s_lab, _ = soft_labels(rows, head_blob["lambda"])
    p = head.logp(rows["z"], rows["phi"]).exp()
    share_p = float((p.max(-1).values >= 0.999).double().mean())
    share_s = float((s_lab.max(-1).values >= 0.999).double().mean())
    lo_lvl, hi_lvl = bracket_levels(h_curve)
    ovp = result["calibration"]["head"]["online_vs_predicted"]
    tech = {"lbfgs_grad_inf": lam_row["fit"]["grad_inf"],
            "lbfgs_converged": lam_row["fit"]["grad_inf"] < 1e-5,
            "sharpness": {"share_max_p_ge_0.999": share_p, "share_max_s_ge_0.999": share_s,
                          "passed": share_p <= share_s + 0.05},
            "calibration_resolution": {"level_below": lo_lvl, "level_above": hi_lvl,
                                       "passed": lo_lvl is not None and hi_lvl is not None and hi_lvl - lo_lvl <= 0.02},
            "online_vs_predicted": {"max_abs": max(abs(x["online"] - x["predicted"]) for x in ovp),
                                    "passed": all(abs(x["online"] - x["predicted"]) <= 0.01 for x in ovp)},
            "brackets_found": h_br is not None and r_br is not None}
    tech["passed"] = (tech["lbfgs_converged"] and tech["sharpness"]["passed"] and tech["calibration_resolution"]["passed"]
                      and tech["online_vs_predicted"]["passed"] and tech["brackets_found"])
    result["technical_checks"] = tech
    print("technical", json.dumps(tech), flush=True)
    if not tech["brackets_found"]:
        result["decision"] = "fail"
        result["gate"] = {"technical": False}
        write_json(RESULTS / "stage1.json", receipt("chunk_depth_v3_2_stage1", **result))
        print("no bracket on calibration; stage 1 stops", flush=True)
        return 1

    # gate on stage1_screen
    r_ev = S.evaluate_aligned(model, screen, lambda nu: S.position_policy(ridge, nu), r_br)
    rec = []
    hi, lo, w = h_br
    h_hi = S.run_all(model, screen, head_policy(head, hi, rec))
    h_lo = S.run_all(model, screen, head_policy(head, lo))
    doc, cost = S.mixed(h_hi, h_lo, w)
    sh_hi, sh_lo = S.summary(h_hi)["depth_share"], S.summary(h_lo)["depth_share"]
    h_ev = {"doc": doc, "cost": cost, "share": {d: w * sh_hi[d] + (1 - w) * sh_lo[d] for d in R.V3},
            "points": {"high": S.summary(h_hi), "low": S.summary(h_lo)}}
    g = R.bootstrap([a - b for a, b in zip(r_ev["doc"], h_ev["doc"])])
    ratio = h_ev["cost"] / r_ev["cost"]
    gate = {"technical": tech["passed"], "cost_ratio": 0.98 <= ratio <= 1.02,
            "effect": g["mean"] >= 0.005 and g["lower_one_sided_90"] > 0}
    result["screen"] = {"R": S.report(r_ev), "head": S.report(h_ev), "g_head": g, "cost_ratio_head_over_R": ratio}
    result["gate"] = gate
    result["decision"] = "pass" if all(gate.values()) else "fail"
    torch.save({"R": r_ev["doc"], "head": h_ev["doc"], "head_high": [s["mean_nll"] for s in h_hi],
                "head_low": [s["mean_nll"] for s in h_lo]}, OUT / "stage1_per_document.pt")
    write_json(RESULTS / "stage1.json", receipt("chunk_depth_v3_2_stage1", **result))
    print(json.dumps({"decision": result["decision"], "gate": gate, "g_head": g, "cost_ratio": ratio,
                      "R": result["screen"]["R"]["doc_mean_nll"], "head": result["screen"]["head"]["doc_mean_nll"]},
                     indent=1), flush=True)

    # diagnostics (not part of the gate)
    diag = {}
    scr_tabs = torch.load(OUT / "tables_stage1_screen.pt")
    agree = n = 0
    it = iter(rec)
    for item, t, s in zip(screen, scr_tabs, h_hi):
        st = doc_structure(item)
        dprev = 16
        for i, d in enumerate(s["depths"]):
            k = int((head.logp(t["reps"][R.V3.index(dprev), i], torch.tensor(st[i]["phi"])) - hi * R.R_REL).argmax())
            agree += int(k == next(it))
            n += 1
            dprev = d
    diag["action_agreement_table_vs_online"] = {"b": hi, "decisions": n, "agreement": agree / n}
    srows = build_rows(screen, scr_tabs, kappa_s, kappa_e)
    s_scr, _ = soft_labels(srows, head_blob["lambda"])
    p_scr = head.logp(srows["z"], srows["phi"]).exp()
    diag["screen_table_fit"] = {
        "weighted_kl_s_p": float((srows["w"] * (s_scr * (s_scr.clamp_min(1e-300).log() - p_scr.log())).sum(-1)).sum()
                                 / srows["w"].sum()),
        "share_max_p_ge_0.999": float((p_scr.max(-1).values >= 0.999).double().mean()),
        "share_max_s_ge_0.999": float((s_scr.max(-1).values >= 0.999).double().mean()),
        "argmax_agreement_p_vs_s": float((p_scr.argmax(-1) == s_scr.argmax(-1)).double().mean())}
    # uniform depths under theta_S, exact calibration costs
    f_cal = [[l["f"] for l in doc_structure(item)] for item in cal]
    rel = {d: sum(float(f[k]) for doc_ in f_cal for f in doc_) / sum(float(f[4]) for doc_ in f_cal for f in doc_)
           for k, d in enumerate(R.V3)}
    d_hi = min(d for d in R.V3 if rel[d] >= S.BUDGET)
    d_lo = max(d for d in R.V3 if rel[d] <= S.BUDGET)
    u_br = (d_hi, d_lo, (S.BUDGET - rel[d_lo]) / (rel[d_hi] - rel[d_lo]))
    u_ev = S.evaluate_aligned(model, screen, S.uniform_policy, u_br)
    diag["uniform"] = {"mixture": list(u_br), "screen": S.report(u_ev),
                       "R_over_uniform": R.bootstrap([a - b for a, b in zip(u_ev["doc"], r_ev["doc"])]),
                       "head_over_uniform": R.bootstrap([a - b for a, b in zip(u_ev["doc"], h_ev["doc"])])}
    print("uniform", S.report(u_ev)["doc_mean_nll"], flush=True)
    # representation ridge (phase-0 recipe) under theta_S
    p_scan, p_br, p_curve = S.calibrate(model, cal, lambda nu: S.rep_ridge_policy(probe, nu),
                                        scores_rep_ridge(cal, cal_tabs, probe), S.NU_FINE, "rep-ridge", timing)
    diag["rep_ridge"] = S.scan_record(p_scan, p_br, p_curve)
    if p_br:
        p_ev = S.evaluate_aligned(model, screen, lambda nu: S.rep_ridge_policy(probe, nu), p_br)
        diag["rep_ridge"].update({"screen": S.report(p_ev),
                                  "R_minus_rep_ridge": R.bootstrap([a - b for a, b in zip(r_ev["doc"], p_ev["doc"])]),
                                  "rep_ridge_minus_head": R.bootstrap([a - b for a, b in zip(p_ev["doc"], h_ev["doc"])])})
    # development set: the same calibrated head and R
    dev = load_role("development")
    r_dev = S.evaluate_aligned(model, dev, lambda nu: S.position_policy(ridge, nu), r_br)
    h_dev = S.evaluate_aligned(model, dev, lambda b: head_policy(head, b), h_br)
    diag["development"] = {"documents": len(dev), "R": S.report(r_dev), "head": S.report(h_dev),
                           "g_head": R.bootstrap([a - b for a, b in zip(r_dev["doc"], h_dev["doc"])]),
                           "cost_ratio": h_dev["cost"] / r_dev["cost"]}
    result["diagnostics"] = diag
    result["seconds_per_online_run"] = sum(timing) / len(timing)
    result["seconds"] = time.perf_counter() - t_start
    write_json(RESULTS / "stage1.json", receipt("chunk_depth_v3_2_stage1", **result))
    print(json.dumps({"agreement": diag["action_agreement_table_vs_online"], "screen_fit": diag["screen_table_fit"],
                      "development_g_head": diag["development"]["g_head"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
