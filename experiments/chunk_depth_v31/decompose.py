"""Pilot diagnostic: split E - R on pilot_screen into a generation (LoRA) part and a controller part.

E-static is the E LoRA with its own position schedule (the pool member of section 6.1). With paired
per-document values on pilot_screen:
  E - R = (E - E_static) + (E_static - R)
the first term compares two controllers on the same LoRA, the second two LoRAs under the same schedule
family. Parameters and mixture weights are read from results/chunk-depth-v31/screen.json (fixed on
pilot_cal); R and E are re-executed and must reproduce the values there.
Writes results/chunk-depth-v31/decompose.json and artifacts/chunk-depth-v31/screen_per_document.pt.
"""

from __future__ import annotations

import json

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v3.phase0_online import Ridge
from experiments.chunk_depth_v31.data import OUT, RESULTS
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31 import screen as S


def position_ridge(model, items, feats, alpha):
    tabs = S.tables(model, items, feats)
    x, y = [], []
    for t in tabs:
        for i, l in enumerate(t["st"]):
            if l["n"]:
                x.append(l["phi"]); y.append(t["line_nll"][i])
    return Ridge(torch.tensor(x, dtype=torch.float64), torch.stack(y), alpha)


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    sc = json.loads((RESULTS / "screen.json").read_text(encoding="utf-8"))
    screen = torch.load(OUT / "items_pilot_screen.pt")
    train_items = torch.load(OUT / "items_pilot_train.pt")
    train_feats = torch.load(OUT / "features_pilot_train.pt")
    model = H.load_model()
    lora = S.Lora(model)
    blob = torch.load(OUT / "readouts.pt")
    readout = S.new_readout()
    readout.load_state_dict(blob["readouts"][blob["selected_lambda"]])
    pool = sc["R_selection"]["pool"]
    ck = {"E-static": OUT / "pilot_E_s17.pt", "SCHED-A": OUT / "pilot_SCHED-A_s17.pt"}
    br = lambda b: (b["param_high_cost"], b["param_low_cost"], b["weight_high"])
    ev = {}
    for name in ("SCHED-A", "E-static"):
        lora.load(ck[name])
        ridge = position_ridge(model, train_items, train_feats, pool[name]["ridge_alpha"])
        ev[name] = S.evaluate_aligned(model, screen, lambda nu, r=ridge: S.position_policy(r, nu), br(pool[name]["bracket"]))
        print(name, S.report(ev[name])["doc_mean_nll"], ev[name]["cost"], flush=True)
    lora.load(ck["E-static"])
    ev["E"] = S.evaluate_aligned(model, screen, lambda b: S.readout_policy(readout, b), br(sc["E_calibration"]["bracket"]))
    print("E", S.report(ev["E"])["doc_mean_nll"], ev["E"]["cost"], flush=True)
    r, e, es = ev["SCHED-A"]["doc"], ev["E"]["doc"], ev["E-static"]["doc"]
    reproduced = {"R": [sum(r) / len(r), sc["screen"]["R"]["doc_mean_nll"]],
                  "E": [sum(e) / len(e), sc["screen"]["E"]["doc_mean_nll"]]}
    out = {"documents": len(screen),
           "reproduction": {k: {"rerun": v[0], "screen_json": v[1], "abs_diff": abs(v[0] - v[1])} for k, v in reproduced.items()},
           "policies": {k: S.report(v) for k, v in ev.items()},
           "E_minus_R_gain": R.bootstrap([a - b for a, b in zip(r, e)]),
           "controller_gain_E_over_E_static": R.bootstrap([a - b for a, b in zip(es, e)]),
           "lora_gain_SCHED_A_over_E_LoRA_under_position_schedule": R.bootstrap([a - b for a, b in zip(es, r)])}
    torch.save({k: v["doc"] for k, v in ev.items()}, OUT / "screen_per_document.pt")
    write_json(RESULTS / "decompose.json", receipt("chunk_depth_v3_1_pilot_decompose", **out))
    print(json.dumps({k: out[k] for k in ("reproduction", "E_minus_R_gain", "controller_gain_E_over_E_static",
                                          "lora_gain_SCHED_A_over_E_LoRA_under_position_schedule")}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
