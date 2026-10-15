"""Pilot diagnostic: saturation of the readout distribution on pilot_cal decision positions.

For the frozen model and the E LoRA: norm of the decision representation, logit spread of the readout,
fraction of decisions whose top probability reaches the Dirichlet-mean ceiling, and the log-ratio between
the top and the second action. Writes results/chunk-depth-v31/saturation.json.
"""

from __future__ import annotations

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v31.data import OUT, RESULTS
from experiments.chunk_depth_v31 import screen as S
from experiments.chunk_depth_v31.train import EPS, ETA, K, KAPPA_D, mean_policy


def quantiles(x):
    q = torch.tensor([0.1, 0.5, 0.9], dtype=x.dtype)
    return [float(v) for v in torch.quantile(x, q)]


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    cal = torch.load(OUT / "items_pilot_cal.pt")
    feats = torch.load(OUT / "features_pilot_cal.pt")
    model = H.load_model()
    lora = S.Lora(model)
    blob = torch.load(OUT / "readouts.pt")
    ceiling = (KAPPA_D * ((1 - EPS) + EPS / K) + ETA) / (KAPPA_D + K * ETA)
    floor = (KAPPA_D * EPS / K + ETA) / (KAPPA_D + K * ETA)
    out = {"dirichlet_mean_ceiling": ceiling, "dirichlet_mean_floor": floor,
           "max_log_ratio": float(torch.log(torch.tensor(ceiling / floor))), "models": {}}
    for name, path in (("frozen", None), ("E", OUT / "pilot_E_s17.pt")):
        lora.load(path)
        tabs = S.tables(model, cal, feats)
        x = torch.cat([torch.cat([t["reps"][:, i], torch.tensor(l["phi"]).expand(K, -1)], 1)
                       for t in tabs for i, l in enumerate(t["st"])])            # rows: (line, previous depth)
        res = {}
        for lam, sd in sorted(blob["readouts"].items()):
            readout = S.new_readout()
            readout.load_state_dict(sd)
            with torch.no_grad():
                logits = readout(x).double()
                mu = mean_policy(logits)
            top2 = mu.log().topk(2, -1).values
            res[str(lam)] = {"logit_range_quantiles": quantiles(logits.max(-1).values - logits.min(-1).values),
                             "top_minus_second_log_mu_quantiles": quantiles(top2[:, 0] - top2[:, 1]),
                             "share_top_at_ceiling": float((mu.max(-1).values >= ceiling - 1e-3).double().mean()),
                             "argmax_share": [float((mu.argmax(-1) == k).double().mean()) for k in range(K)],
                             "weight_norm": float(readout.weight.norm())}
        out["models"][name] = {"rows": len(x), "rep_norm_quantiles": quantiles(x[:, :5280].double().norm(dim=-1)),
                               "readouts": res}
        print(name, out["models"][name]["rep_norm_quantiles"], res[str(blob["selected_lambda"])], flush=True)
    out["selected_lambda"] = blob["selected_lambda"]
    write_json(RESULTS / "saturation.json", receipt("chunk_depth_v3_1_pilot_saturation", **out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
