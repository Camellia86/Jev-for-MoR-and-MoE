"""Pilot diagnostic: 2 x 2 cross evaluation of generation parameters and position schedules on pilot_screen.

theta_S, theta_E are the SCHED-A and E LoRA parameters; pi_S, pi_E the position schedules fitted on
pilot_train tables under each of them (ridge alpha and the budget-aligned nu pair with its weight are read
from results/chunk-depth-v31/screen.json, fixed on pilot_cal). A position schedule chooses depths from the
line index and the previous line length only, so a schedule gives every document the same depths whatever
the generation parameters; the run asserts this, and the cost of each document is then identical across rows.

  L(theta_E, pi_E) - L(theta_S, pi_S)
    = [L(theta_E, pi_S) - L(theta_S, pi_S)] + [L(theta_E, pi_E) - L(theta_E, pi_S)]      (path 1)
    = [L(theta_E, pi_E) - L(theta_S, pi_E)] + [L(theta_S, pi_E) - L(theta_S, pi_S)]      (path 2)

The two diagonal cells are re-used from artifacts/chunk-depth-v31/screen_per_document.pt (decompose.py).
Writes results/chunk-depth-v31/crosseval.json and artifacts/chunk-depth-v31/crosseval_per_document.pt.
"""

from __future__ import annotations

import json

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v31.data import OUT, RESULTS
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31 import screen as S
from experiments.chunk_depth_v31.decompose import position_ridge
from experiments.chunk_depth_v31.train import doc_structure


def planned_depths(items, ridge, nu):
    out = []
    for item in items:
        st = doc_structure(item)
        x = torch.tensor([l["phi"] for l in st], dtype=torch.float64)
        out.append([R.V3[int(k)] for k in (ridge(x) + nu * R.R_REL).argmin(1)])
    return out


def run_cell(model, items, ridge, br):
    hi, lo, w = br
    cells = {}
    for nu in (hi, lo):
        stats = S.run_all(model, items, S.position_policy(ridge, nu))
        assert [s["depths"] for s in stats] == planned_depths(items, ridge, nu), "schedule depends on model state"
        cells[nu] = stats
    doc = [w * a["mean_nll"] + (1 - w) * b["mean_nll"] for a, b in zip(cells[hi], cells[lo])]
    cost = [w * a["cost"] + (1 - w) * b["cost"] for a, b in zip(cells[hi], cells[lo])]
    base = [a["base16"] for a in cells[hi]]
    return doc, cost, base


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    sc = json.loads((RESULTS / "screen.json").read_text(encoding="utf-8"))
    pool = sc["R_selection"]["pool"]
    screen = torch.load(OUT / "items_pilot_screen.pt")
    train_items = torch.load(OUT / "items_pilot_train.pt")
    train_feats = torch.load(OUT / "features_pilot_train.pt")
    model = H.load_model()
    lora = S.Lora(model)
    theta = {"S": OUT / "pilot_SCHED-A_s17.pt", "E": OUT / "pilot_E_s17.pt"}
    pool_name = {"S": "SCHED-A", "E": "E-static"}
    br = {k: (pool[v]["bracket"]["param_high_cost"], pool[v]["bracket"]["param_low_cost"], pool[v]["bracket"]["weight_high"])
          for k, v in pool_name.items()}
    ridges = {}
    for k in ("S", "E"):
        lora.load(theta[k])
        ridges[k] = position_ridge(model, train_items, train_feats, pool[pool_name[k]]["ridge_alpha"])
    del train_feats
    diag = torch.load(OUT / "screen_per_document.pt")
    L = {("S", "S"): diag["SCHED-A"], ("E", "E"): diag["E-static"]}
    cost = {}
    for th, pi in (("E", "S"), ("S", "E")):
        lora.load(theta[th])
        L[(th, pi)], cost[(th, pi)], base = run_cell(model, screen, ridges[pi], br[pi])
        print(f"theta_{th} pi_{pi}", sum(L[(th, pi)]) / len(screen), sum(cost[(th, pi)]) / sum(base), flush=True)
    # the diagonal cells must reproduce screen.json / decompose.json
    dec = json.loads((RESULTS / "decompose.json").read_text(encoding="utf-8"))["policies"]
    reproduced = {"theta_S_pi_S": [sum(L[("S", "S")]) / len(screen), dec["SCHED-A"]["doc_mean_nll"]],
                  "theta_E_pi_E": [sum(L[("E", "E")]) / len(screen), dec["E-static"]["doc_mean_nll"]]}

    def diff(a, b):
        return R.bootstrap([x - y for x, y in zip(L[a], L[b])])

    out = {
        "documents": len(screen),
        "sign": "loss differences in nat/token; positive means the first term has higher NLL",
        "schedules": {k: {"ridge_alpha": pool[v]["ridge_alpha"], "nu_high_cost": br[k][0], "nu_low_cost": br[k][1],
                          "weight_high": br[k][2]} for k, v in pool_name.items()},
        "relative_cost": {f"pi_{pi}": sum(cost[(th, pi)]) / sum(base) for th, pi in cost},
        "table": {f"theta_{th}_pi_{pi}": sum(v) / len(v) for (th, pi), v in sorted(L.items())},
        "diagonal_reproduction": {k: {"value": v[0], "decompose_json": v[1], "abs_diff": abs(v[0] - v[1])}
                                  for k, v in reproduced.items()},
        "total_EE_minus_SS": diff(("E", "E"), ("S", "S")),
        "path1_parameters_at_pi_S": diff(("E", "S"), ("S", "S")),
        "path1_schedule_at_theta_E": diff(("E", "E"), ("E", "S")),
        "path2_parameters_at_pi_E": diff(("E", "E"), ("S", "E")),
        "path2_schedule_at_theta_S": diff(("S", "E"), ("S", "S")),
        "interaction": R.bootstrap([(a - b) - (c - d) for a, b, c, d in
                                    zip(L[("E", "E")], L[("S", "E")], L[("E", "S")], L[("S", "S")])]),
    }
    torch.save({f"theta_{th}_pi_{pi}": v for (th, pi), v in L.items()}, OUT / "crosseval_per_document.pt")
    write_json(RESULTS / "crosseval.json", receipt("chunk_depth_v3_1_pilot_crosseval", **out))
    print(json.dumps({k: out[k] for k in ("relative_cost", "table", "diagonal_reproduction")}, indent=1))
    for k in ("total_EE_minus_SS", "path1_parameters_at_pi_S", "path1_schedule_at_theta_E",
              "path2_parameters_at_pi_E", "path2_schedule_at_theta_S", "interaction"):
        print(k, round(out[k]["mean"], 5), [round(x, 5) for x in out[k]["ci95"]])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
