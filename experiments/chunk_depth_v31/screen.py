"""Pilot step 5 (plan v3.1 sections 6.1-6.5, 7.2, 7.4): R selection, E calibration, screening and go/no-go.

Budget alignment (section 3.4): two parameters whose real-execution pilot_cal costs bracket the budget are
mixed at document level; the weight is fixed on pilot_cal and applied unchanged on pilot_screen.

Which parameters to execute is found on a fine grid first. Every rule here has the form
k = argmax_k [score_c(k | previous depth) - p * r(k)]; its cost on pilot_cal is predicted from table
scores (the position schedule and uniform depths have exact predicted costs, since their decisions do not
depend on the model state; readout and probe rules use the table representation z-tilde). The two
parameters nearest the budget from above and below are then executed online; if their online costs do
not bracket the budget, the search moves outward one predicted cost level at a time.

Writes results/chunk-depth-v31/screen.json (the gate part is written before the diagnostics start).
"""

from __future__ import annotations

import argparse
import json
import re
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import MODEL_DIR, receipt, write_json
from experiments.chunk_depth_v3.phase0_online import fit_ridge
from experiments.chunk_depth_v31.data import OUT, RESULTS, state_matrix
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31.train import K, doc_structure, mean_policy, new_readout

BUDGET = 0.5
NU_FINE = [round(i * 0.01, 2) for i in range(0, 401)]
B_FINE = [round(-20 + i * 0.25, 2) for i in range(0, 481)]
EOS = 65505
A10_FACTOR = 2.0          # assumed A10 / RTX 5080 time ratio (memory bandwidth 600 vs 960 GB/s, rounded up)
CAPS = {"features_h": 16, "training_h": 24, "evaluation_h": 40, "generation_h": 44}


# ----------------------------------------------------------------------------- model state

class Lora:
    def __init__(self, model):
        self.modules = H.attach_lora(model, 8, 16, seed=0)
        self.params = H.lora_parameters(self.modules)

    def load(self, path):
        if path is None:
            H.set_lora_enabled(self.modules, False)
            return
        H.set_lora_enabled(self.modules, True)
        blob = torch.load(path)
        with torch.no_grad():
            for p, v in zip(self.params, blob["lora"]):
                p.copy_(v.cuda())


@torch.no_grad()
def tables(model, items, feats):
    """Per line: per-token table NLL at each depth [lines, K]; decision representations [K, lines, 5280]
    with every earlier position at the same depth (the training representation z-tilde)."""
    out = []
    for item, f in zip(items, feats):
        S = len(item["ids"])
        ids = torch.tensor(item["ids"], device="cuda")
        fc = H.freqs(model, torch.arange(S))
        st = doc_structure(item)
        nll, reps = [], []
        for k in range(K):
            z = H.coda(model, f["feats"][k].cuda()[None], fc)[0]
            reps.append(z[[l["a"] - 1 for l in st]].float().cpu())
            nll.append(torch.nn.functional.cross_entropy(H.head(model, z[None])[0, :-1].float(), ids[1:],
                                                         reduction="none").cpu())
        nll = torch.stack(nll).double()
        per_line = torch.stack([nll[:, l["scored"]].mean(1) if l["n"] else torch.full((K,), float("nan"),
                                dtype=torch.float64) for l in st])
        out.append({"st": st, "line_nll": per_line, "reps": torch.stack(reps)})
    return out


# ----------------------------------------------------------------------------- policies

def position_policy(ridge, nu):
    def pol(z, dz, c, prev_len, hist):
        x = torch.tensor([R.pos_feats(c, prev_len)], dtype=torch.float64)
        return R.V3[int((ridge(x)[0] + nu * R.R_REL).argmin())]
    return pol


def readout_choice(readout, x, b):
    with torch.no_grad():
        mu = mean_policy(readout(x[None]))[0]
    return int((mu.log() - b * R.R_REL).argmax())


def readout_policy(readout, b, record=None):
    def pol(z, dz, c, prev_len, hist):
        k = readout_choice(readout, torch.cat([z.float(), torch.tensor(R.pos_feats(c, prev_len))]), b)
        if record is not None:
            record.append(k)
        return R.V3[k]
    return pol


def table_rep_policy(readout, b, reps):
    """Same rule, but the representation is z-tilde: the table run at the previous line's online depth."""
    def pol(z, dz, c, prev_len, hist):
        x = torch.cat([reps[R.V3.index(dz), c - 1], torch.tensor(R.pos_feats(c, prev_len))])
        return R.V3[readout_choice(readout, x, b)]
    return pol


def probe_x(z, kprev_onehot, phi):
    return torch.cat([z.double(), kprev_onehot, phi.double()], -1)


def rep_ridge_policy(ridge, nu):
    def pol(z, dz, c, prev_len, hist):
        onehot = torch.tensor([1.0 if dz == d else 0.0 for d in R.V3], dtype=torch.float64)
        x = probe_x(z, onehot, torch.tensor(R.pos_feats(c, prev_len)))[None]
        return R.V3[int((ridge(x)[0] + nu * R.R_REL).argmin())]
    return pol


def uniform_policy(d):
    return lambda z, dz, c, prev_len, hist: d


# ----------------------------------------------------------------------------- table scores and cost prediction

def scores_position(items, ridge):
    out = []
    for item in items:
        st = doc_structure(item)
        s = -ridge(torch.tensor([l["phi"] for l in st], dtype=torch.float64))          # [lines, K]
        out.append({"s": [row.expand(K, -1) for row in s], "f": [l["f"] for l in st]})
    return out


@torch.no_grad()
def scores_readout(tabs, readout):
    out = []
    for t in tabs:
        rows = []
        for i, l in enumerate(t["st"]):
            x = torch.cat([t["reps"][:, i], torch.tensor(l["phi"]).expand(K, -1)], 1)    # [K_prev, 5290]
            rows.append(mean_policy(readout(x)).log())
        out.append({"s": rows, "f": [l["f"] for l in t["st"]]})
    return out


def scores_probe(tabs, probe):
    out = []
    for t in tabs:
        rows = []
        for i, l in enumerate(t["st"]):
            x = probe_x(t["reps"][:, i], torch.eye(K, dtype=torch.float64),
                        torch.tensor(l["phi"], dtype=torch.float64).expand(K, -1))
            rows.append(-probe(x))
        out.append({"s": rows, "f": [l["f"] for l in t["st"]]})
    return out


def predicted_cost(scores, p):
    """Sequential decisions on table scores: line 1 follows the question at depth 16."""
    num = den = 0.0
    for doc in scores:
        kprev = 4
        for s, f in zip(doc["s"], doc["f"]):
            k = int((s[kprev] - p * R.R_REL).argmax())
            num += float(f[k])
            den += float(f[4])
            kprev = k
    return num / den


# ----------------------------------------------------------------------------- execution and alignment

def run_all(model, items, policy, record=None, per_document=False):
    """policy: a policy, or (per_document=True) a function item_index -> policy.
    record (decisions appended by the policy) is rolled back to the document start on an OOM retry."""
    out = []
    for j, item in enumerate(items):
        pol = policy(j) if per_document else policy
        mark = None if record is None else len(record)

        def once():
            if record is not None:
                del record[mark:]
            return R.online(model, item, pol)
        nll, depths = R.retry(once)
        out.append(R.doc_stats(item, nll, depths))
    return out


def summary(stats):
    shares = [sum(x == d for s in stats for x in s["depths"]) for d in R.V3]
    return {"relative_cost": sum(s["cost"] for s in stats) / sum(s["base16"] for s in stats),
            "doc_mean_nll": sum(s["mean_nll"] for s in stats) / len(stats),
            "depth_share": {d: n / sum(shares) for d, n in zip(R.V3, shares)}}


def bracket(scan):
    """scan: {param: stats}. Adjacent points in cost order straddling the budget -> (p_hi, p_lo, w_hi)."""
    pts = sorted((summary(s)["relative_cost"], p) for p, s in scan.items())
    for (c0, p0), (c1, p1) in zip(pts, pts[1:]):
        if c0 <= BUDGET <= c1 and c1 > c0:
            return p1, p0, (BUDGET - c0) / (c1 - c0)
    return None


def calibrate(model, items, make, scores, fine, label, timing, max_moves=6):
    pred = {p: predicted_cost(scores, p) for p in fine}
    levels = {}
    for p, c in pred.items():
        levels.setdefault(round(c, 10), []).append(p)
    lv = sorted(levels)                                       # ascending predicted cost
    rep = {c: sorted(ps)[len(ps) // 2] for c, ps in levels.items()}
    above = [i for i, c in enumerate(lv) if c >= BUDGET]
    below = [i for i, c in enumerate(lv) if c <= BUDGET]
    i_hi = above[0] if above else len(lv) - 1
    i_lo = below[-1] if below else 0
    scan = {}
    for _ in range(max_moves + 1):
        for i in sorted({i_hi, i_lo}):
            p = rep[lv[i]]
            if p in scan:
                continue
            t0 = time.perf_counter()
            scan[p] = run_all(model, items, make(p))
            timing.append((time.perf_counter() - t0) / len(items))
            s = summary(scan[p])
            print(label, p, "pred", round(pred[p], 4), "online", round(s["relative_cost"], 4),
                  round(s["doc_mean_nll"], 4), flush=True)
        if bracket(scan):
            break
        if min(summary(s)["relative_cost"] for s in scan.values()) > BUDGET:
            if i_lo == 0:
                break
            i_lo -= 1
        else:
            if i_hi == len(lv) - 1:
                break
            i_hi += 1
    br = bracket(scan)
    curve = [{"param": rep[c], "param_range": [min(levels[c]), max(levels[c])], "predicted_cost": c} for c in lv]
    return scan, br, curve


def mixed(stats_hi, stats_lo, w):
    doc = [w * a["mean_nll"] + (1 - w) * b["mean_nll"] for a, b in zip(stats_hi, stats_lo)]
    cost = sum(w * a["cost"] + (1 - w) * b["cost"] for a, b in zip(stats_hi, stats_lo))
    return doc, cost / sum(a["base16"] for a in stats_hi)


def scan_record(scan, br, curve=None):
    out = {"online_points": [{"param": p, **summary(s)} for p, s in sorted(scan.items())],
           "bracket": None if br is None else {"param_high_cost": br[0], "param_low_cost": br[1], "weight_high": br[2]},
           "cal_matched_nll": None if br is None else sum(mixed(scan[br[0]], scan[br[1]], br[2])[0]) / len(scan[br[0]])}
    if curve is not None:
        near = [x for x in curve if abs(x["predicted_cost"] - BUDGET) <= 0.15]
        out["predicted_levels_near_budget"] = near
        out["predicted_levels"] = len(curve)
    return out


def evaluate_aligned(model, items, make, br):
    hi, lo, w = br
    s_hi, s_lo = run_all(model, items, make(hi)), run_all(model, items, make(lo))
    doc, cost = mixed(s_hi, s_lo, w)
    sh_hi, sh_lo = summary(s_hi)["depth_share"], summary(s_lo)["depth_share"]
    return {"doc": doc, "cost": cost, "share": {d: w * sh_hi[d] + (1 - w) * sh_lo[d] for d in R.V3},
            "points": {"high": summary(s_hi), "low": summary(s_lo)}}


def report(ev):
    return {"doc_mean_nll": sum(ev["doc"]) / len(ev["doc"]), "relative_cost": ev["cost"], "depth_share": ev["share"],
            "components": ev["points"]}


# ----------------------------------------------------------------------------- generation floor check

def last_number(text):
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return nums[-1].replace(",", "").rstrip(".") if nums else ""


@torch.no_grad()
def generate(model, tok, item, depth=16, max_new=256):
    q_end, _, _ = R.lines_of(item)
    state = state_matrix(item["split"], item["row"], q_end + max_new).cuda()[None]
    cache = R.new_cache()
    ids = torch.tensor(item["ids"][:q_end], device="cuda")[None]
    _, logits = R.run_segment(model, cache, ids, state, 0, q_end, depth)
    out = []
    nxt = int(logits[-1].argmax())
    for t in range(q_end, q_end + max_new):
        if nxt == EOS:
            break
        out.append(nxt)
        if "\nQuestion:" in tok.decode(out[-8:]):
            break
        ids = torch.cat([ids, torch.tensor([[nxt]], device="cuda")], 1)
        _, logits = R.run_segment(model, cache, ids, state, t, t + 1, depth)
        nxt = int(logits[-1].argmax())
    return tok.decode(out).split("\nQuestion:")[0]


def generation_floor(model, items, n=32):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    correct, samples, t0 = 0, [], time.perf_counter()
    for item in items[:n]:
        text = R.retry(generate, model, tok, item)
        pred = last_number(text)
        try:
            ok = abs(float(pred) - float(item["gold"])) < 1e-6
        except ValueError:
            ok = False
        correct += int(ok)
        samples.append({"gold": item["gold"], "pred": pred, "ok": ok, "text": text[-200:]})
    return {"problems": n, "depth": 16, "accuracy": correct / n, "seconds_per_problem": (time.perf_counter() - t0) / n,
            "samples": samples[:6]}


# ----------------------------------------------------------------------------- main

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--skip-diagnostics", action="store_true")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    t_start = time.perf_counter()
    cal = torch.load(OUT / "items_pilot_cal.pt")
    cal_feats = torch.load(OUT / "features_pilot_cal.pt")
    screen = torch.load(OUT / "items_pilot_screen.pt")
    train_items = torch.load(OUT / "items_pilot_train.pt")
    train_feats = torch.load(OUT / "features_pilot_train.pt")
    model = H.load_model()
    lora = Lora(model)
    ck = {name: OUT / f"pilot_{name}_s{args.seed}.pt" for name in ("E", "J", "SCHED-A", "FMIX-A")}
    blob = torch.load(OUT / "readouts.pt")
    readout = new_readout()
    readout.load_state_dict(blob["readouts"][blob["selected_lambda"]])
    timing, table_timing = [], []
    result = {"budget": BUDGET, "seed": args.seed,
              "documents": {"pilot_train": len(train_items), "pilot_cal": len(cal), "pilot_screen": len(screen)},
              "parameter_search": "fine-grid cost prediction from table scores, then online execution of the "
                                  "parameters nearest the budget; alignment and selection use online results only"}

    # 1. R: one position schedule family per pool member, fitted on pilot_train tables, chosen on pilot_cal
    pool = {"frozen": None, "SCHED-A": ck["SCHED-A"], "FMIX-A": ck["FMIX-A"], "E-static": ck["E"]}
    pool_res, ridges, brs, frozen_tabs = {}, {}, {}, None
    for name, path in pool.items():
        lora.load(path)
        t0 = time.perf_counter()
        tabs = tables(model, train_items, train_feats)
        table_timing.append((time.perf_counter() - t0) / len(train_items))
        if name == "frozen":
            frozen_tabs = tabs
        x, y, g = [], [], []
        for j, t in enumerate(tabs):
            for i, l in enumerate(t["st"]):
                if l["n"]:
                    x.append(l["phi"]); y.append(t["line_nll"][i]); g.append(j)
        ridge, alpha = fit_ridge(torch.tensor(x, dtype=torch.float64), torch.stack(y), torch.tensor(g),
                                 (0.01, 0.1, 1.0, 10.0, 100.0))
        ridges[name] = ridge
        scan, brs[name], curve = calibrate(model, cal, lambda nu, r=ridge: position_policy(r, nu),
                                           scores_position(cal, ridge), NU_FINE, f"R:{name}", timing)
        pool_res[name] = {"ridge_alpha": alpha, **scan_record(scan, brs[name], curve)}
    del train_feats
    best = min((n for n in pool_res if brs[n]), key=lambda n: pool_res[n]["cal_matched_nll"])
    result["R_selection"] = {"pool": pool_res, "selected": best}
    print("R selected", best, pool_res[best]["cal_matched_nll"], flush=True)

    # 2. E calibration on pilot_cal
    lora.load(ck["E"])
    e_scan, e_br, e_curve = calibrate(model, cal, lambda b: readout_policy(readout, b),
                                      scores_readout(tables(model, cal, cal_feats), readout), B_FINE, "E", timing)
    result["E_calibration"] = {"lambda": blob["selected_lambda"], **scan_record(e_scan, e_br, e_curve)}

    # 3. screening on pilot_screen
    lora.load(pool[best])
    r_ev = evaluate_aligned(model, screen, lambda nu: position_policy(ridges[best], nu), brs[best])
    lora.load(ck["E"])
    e_ev = evaluate_aligned(model, screen, lambda b: readout_policy(readout, b), e_br)
    gain = R.bootstrap([r - e for r, e in zip(r_ev["doc"], e_ev["doc"])])
    gate = gain["mean"] >= 0.005 and gain["lower_one_sided_90"] > 0
    result["screen"] = {"R": report(r_ev), "E": report(e_ev), "E_gain_over_R": gain,
                        "cost_ratio_E_over_R": e_ev["cost"] / r_ev["cost"], "gate_E_vs_R": gate}

    # 4. training versus deployment representation (section 7.2), at the higher-cost calibrated b
    screen_feats = torch.load(OUT / "features_pilot_screen.pt")
    screen_tabs = tables(model, screen, screen_feats)
    del screen_feats
    b_hi = e_br[0]
    rec_online = []
    online_stats = run_all(model, screen, readout_policy(readout, b_hi, rec_online))
    agree = n_dec = 0
    it = iter(rec_online)
    for t, s in zip(screen_tabs, online_stats):
        dprev = 16
        for i, d in enumerate(s["depths"]):
            x = torch.cat([t["reps"][R.V3.index(dprev), i], torch.tensor(t["st"][i]["phi"])])
            agree += int(readout_choice(readout, x, b_hi) == next(it))
            n_dec += 1
            dprev = d
    tab_stats = run_all(model, screen, lambda j: table_rep_policy(readout, b_hi, screen_tabs[j]["reps"]),
                        per_document=True)
    del screen_tabs
    result["representation_gap"] = {
        "b": b_hi, "decisions": n_dec, "action_agreement_same_history": agree / n_dec,
        "online_z": summary(online_stats), "table_z_tilde": summary(tab_stats),
        "nll_table_minus_online": R.bootstrap([a["mean_nll"] - b["mean_nll"] for a, b in zip(tab_stats, online_stats)])}

    checks = json.loads((RESULTS / "checks.json").read_text(encoding="utf-8"))["status"] == "passed"
    result["go_partial"] = {"technical_checks": checks, "E_vs_R": gate}
    write_json(RESULTS / "screen.json", receipt("chunk_depth_v3_1_pilot_screen", **result))
    print(json.dumps({"screen": result["screen"],
                      "representation_gap": {k: result["representation_gap"][k]
                                             for k in ("action_agreement_same_history", "nll_table_minus_online")}},
                     indent=1, default=str), flush=True)

    # 5. diagnostics
    gen_s = None
    if not args.skip_diagnostics:
        diag = {}
        lora.load(None)
        # uniform frozen depths: costs are exact, the two depths around the budget are mixed
        f_cal = [[l["f"] for l in doc_structure(item)] for item in cal]
        rel = {d: sum(float(f[k]) for doc in f_cal for f in doc) / sum(float(f[4]) for doc in f_cal for f in doc)
               for k, d in enumerate(R.V3)}
        d_hi = min(d for d in R.V3 if rel[d] >= BUDGET)
        d_lo = max(d for d in R.V3 if rel[d] <= BUDGET)
        u_br = (d_hi, d_lo, (BUDGET - rel[d_lo]) / (rel[d_hi] - rel[d_lo]))
        u_ev = evaluate_aligned(model, screen, uniform_policy, u_br)
        diag["uniform_frozen"] = {"cal_cost": rel, "mixture": list(u_br), "screen": report(u_ev),
                                  "R_gain_over_uniform": R.bootstrap([u - r for u, r in zip(u_ev["doc"], r_ev["doc"])]),
                                  "E_gain_over_uniform": R.bootstrap([u - e for u, e in zip(u_ev["doc"], e_ev["doc"])])}
        print("uniform", json.dumps(report(u_ev)["doc_mean_nll"]), flush=True)
        # frozen representation probe (phase-0 recipe) on pilot_train tables of the frozen model
        xr, xp, y, g = [], [], [], []
        for j, t in enumerate(frozen_tabs):
            for i, l in enumerate(t["st"]):
                if not l["n"]:
                    continue
                for k in (range(K) if i > 0 else (4,)):
                    onehot = torch.tensor([1.0 if k == kk else 0.0 for kk in range(K)], dtype=torch.float64)
                    xr.append(probe_x(t["reps"][k, i], onehot, torch.tensor(l["phi"], dtype=torch.float64)))
                    y.append(t["line_nll"][i]); g.append(j)
        del frozen_tabs
        probe, probe_alpha = fit_ridge(torch.stack(xr), torch.stack(y), torch.tensor(g), (1e2, 1e3, 1e4, 1e5))
        del xr
        cal_tabs_frozen = tables(model, cal, cal_feats)
        f_scan, f_br, f_curve = calibrate(model, cal, lambda nu: rep_ridge_policy(probe, nu),
                                          scores_probe(cal_tabs_frozen, probe), NU_FINE, "frozen-probe", timing)
        del cal_tabs_frozen
        diag["frozen_probe"] = {"ridge_alpha": probe_alpha, **scan_record(f_scan, f_br, f_curve)}
        if f_br:
            f_ev = evaluate_aligned(model, screen, lambda nu: rep_ridge_policy(probe, nu), f_br)
            diag["frozen_probe"].update({"screen": report(f_ev),
                                         "probe_gain_over_R": R.bootstrap([r - f for r, f in zip(r_ev["doc"], f_ev["doc"])]),
                                         "E_gain_over_probe": R.bootstrap([f - e for f, e in zip(f_ev["doc"], e_ev["doc"])])})
        # J, the secondary arm, calibrated the same way
        lora.load(ck["J"])
        j_scan, j_br, j_curve = calibrate(model, cal, lambda b: readout_policy(readout, b),
                                          scores_readout(tables(model, cal, cal_feats), readout), B_FINE, "J", timing)
        diag["J"] = scan_record(j_scan, j_br, j_curve)
        if j_br:
            j_ev = evaluate_aligned(model, screen, lambda b: readout_policy(readout, b), j_br)
            diag["J"].update({"screen": report(j_ev),
                              "J_gain_over_R": R.bootstrap([r - x for r, x in zip(r_ev["doc"], j_ev["doc"])]),
                              "E_gain_over_J": R.bootstrap([x - e for x, e in zip(j_ev["doc"], e_ev["doc"])])})
        lora.load(None)
        diag["generation_floor"] = generation_floor(model, screen)
        gen_s = diag["generation_floor"]["seconds_per_problem"]
        result["diagnostics"] = diag

    # 6. resource forecast for section 8 (RTX 5080 timings scaled by A10_FACTOR)
    run_s = sum(timing) / len(timing)
    tab_s = sum(table_timing) / len(table_timing)
    upd_s = json.loads((RESULTS / f"pilot_train_E_s{args.seed}.json").read_text(encoding="utf-8"))["seconds_per_update"]
    n_formal = {"train": 4111, "cal": 389, "screen": 745, "confirm": 1310}
    rtx = {"features_h": sum(n_formal.values()) * 0.21 / 3600,
           "training_h": 10 * 1028 * upd_s / 3600,
           "evaluation_h": (10 * n_formal["train"] * tab_s
                            + n_formal["cal"] * (10 + 3) * 13 * run_s
                            + (n_formal["screen"] + n_formal["confirm"]) * 14 * run_s) / 3600,
           "generation_h": 8 * 512 * (gen_s if gen_s is not None else 20.0) / 3600}
    a10 = {k: v * A10_FACTOR for k, v in rtx.items()}
    result["forecast"] = {"assumptions": {"a10_factor": A10_FACTOR, "seconds_per_online_run": run_s,
                                          "seconds_per_table_document": tab_s, "seconds_per_update": upd_s,
                                          "feature_seconds_per_document": 0.21,
                                          "evaluation_runs": "train tables for 10 models; cal: 10 pool models and "
                                                             "3 E seeds, 13 values each; screen+confirm: 14 runs",
                                          "generation": "8 policies x 512 problems at the depth-16 time per problem"},
                          "rtx5080_h": rtx, "a10_h": a10, "caps_h": CAPS,
                          "within_caps": all(a10[k] <= CAPS[k] for k in CAPS)}
    result["go"] = {"technical_checks": checks, "resource_forecast": result["forecast"]["within_caps"],
                    "E_vs_R": gate}
    result["decision"] = "go" if all(result["go"].values()) else "no_go"
    result["seconds"] = time.perf_counter() - t_start
    write_json(RESULTS / "screen.json", receipt("chunk_depth_v3_1_pilot_screen", **result))
    print(json.dumps({"decision": result["decision"], "go": result["go"], "forecast_a10_h": a10}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
