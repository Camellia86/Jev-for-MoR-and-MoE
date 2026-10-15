"""Execute only the two missing crossed LoRA/schedule evaluations."""
from __future__ import annotations

import json
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.chunk_depth_v31 import run as R, screen as S
from experiments.chunk_depth_v31.data import OUT as OLD_OUT, RESULTS as OLD_RESULTS
from experiments.chunk_depth_v31.decompose import position_ridge
from . import common as C


def main():
    C.verify_inputs()
    torch.backends.cuda.matmul.allow_tf32 = False
    start = time.perf_counter()
    old = json.loads((OLD_RESULTS / "screen.json").read_text(encoding="utf-8"))
    decomp = json.loads((OLD_RESULTS / "decompose.json").read_text(encoding="utf-8"))
    diagonal = torch.load(OLD_OUT / "screen_per_document.pt", map_location="cpu")
    items = torch.load(OLD_OUT / "items_pilot_screen.pt")
    train = torch.load(OLD_OUT / "items_pilot_train.pt")
    feats = torch.load(OLD_OUT / "features_pilot_train.pt")
    model = H.load_model()
    lora = S.Lora(model)
    ridges, brackets = {}, {}
    ck = {"S": OLD_OUT / "pilot_SCHED-A_s17.pt", "E": OLD_OUT / "pilot_E_s17.pt"}
    pool_key = {"S": "SCHED-A", "E": "E-static"}
    for name in ("S", "E"):
        pool = old["R_selection"]["pool"][pool_key[name]]
        lora.load(ck[name])
        ridges[name] = position_ridge(model, train, feats, pool["ridge_alpha"])
        br = pool["bracket"]
        brackets[name] = (br["param_high_cost"], br["param_low_cost"], br["weight_high"])
    del train, feats
    cells = {}
    for name in ("S", "E"):
        key = pool_key[name]
        vals = diagonal[key]
        expected = decomp["policies"][key]["doc_mean_nll"]
        if abs(sum(vals) / len(vals) - expected) > 1e-10:
            raise RuntimeError("Existing diagonal does not reproduce its receipt")
        cells[f"{name}_{name}"] = {"doc": vals, "mean": expected,
                                   "cost": decomp["policies"][key]["relative_cost"],
                                   "source": "existing diagonal, checked against decompose.json"}
    for param, schedule in (("E", "S"), ("S", "E")):
        lora.load(ck[param])
        hi, lo, w = brackets[schedule]
        stats = []
        for nu in (hi, lo):
            rows = []
            policy = S.position_policy(ridges[schedule], nu)
            for j, it in enumerate(items):
                C.check_time(start, 5400)
                nll, depths = R.retry(R.online, model, it, policy)
                rows.append(R.doc_stats(it, nll, depths))
                if (j + 1) % 50 == 0:
                    print(f"cross theta={param} schedule={schedule} nu={nu} {j+1}/{len(items)}", flush=True)
            stats.append(rows)
        vals, cost = S.mixed(stats[0], stats[1], w)
        if abs(cost - cells[f"{schedule}_{schedule}"]["cost"]) > 1e-12:
            raise RuntimeError("A fixed schedule must have identical cost for both LoRAs")
        cells[f"{param}_{schedule}"] = {"doc": vals, "mean": sum(vals)/len(vals), "cost": cost,
                                          "source": "new online execution"}
        torch.save(cells, C.OUT / "cross_cells.pt")
        print(f"cross {param}_{schedule}: mean={sum(vals)/len(vals):.8f} cost={cost:.8f}", flush=True)
    effects = {}
    for schedule in ("S", "E"):
        effects[f"E_minus_S_fixed_schedule_{schedule}"] = R.bootstrap(
            [a-b for a,b in zip(cells[f"E_{schedule}"]["doc"], cells[f"S_{schedule}"]["doc"])])
    for param in ("S", "E"):
        effects[f"schedule_E_minus_S_fixed_theta_{param}"] = R.bootstrap(
            [a-b for a,b in zip(cells[f"{param}_E"]["doc"], cells[f"{param}_S"]["doc"])])
    interaction = [ee-es-se+ss for ee,es,se,ss in zip(cells["E_E"]["doc"],cells["E_S"]["doc"],
                                                      cells["S_E"]["doc"],cells["S_S"]["doc"])]
    effects["interaction"] = R.bootstrap(interaction)
    payload = C.record("cross", documents=len(items),
                       cells={k: {f:v for f,v in x.items() if f != "doc"} for k,x in cells.items()},
                       effects=effects, seconds=time.perf_counter()-start,
                       interpretation="Positive E_minus_S means E has higher NLL at exactly the same schedule and cost.")
    C.write_json(C.RESULTS / "cross.json", payload)
    print(json.dumps(payload["effects"], indent=2), flush=True)


if __name__ == "__main__":
    main()
