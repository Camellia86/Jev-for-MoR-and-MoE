"""Phase 0 addendum: compare online policies at exactly the same expected budget.

Each online policy family is indexed by nu. On cal, take the two scanned nu values whose realised costs
bracket the budget and mix them at the document level with weight w chosen so that the cal cost equals the
budget, exactly as the static mixture mixes two uniform depths. Apply the same nu pair and weight on test.
Reuses the fitted predictors by re-fitting them with the same deterministic procedure as phase0_online.
"""

from __future__ import annotations

import json

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v3 import phase0_online as P
from experiments.chunk_depth_v3.survey import OUT, RESULTS


def bracket(scan, budget):
    pts = sorted(scan, key=lambda s: s["relative_cost"])
    for lo, hi in zip(pts, pts[1:]):
        if lo["relative_cost"] <= budget <= hi["relative_cost"]:
            w = (budget - lo["relative_cost"]) / (hi["relative_cost"] - lo["relative_cost"])
            return lo["nu"], hi["nu"], w
    raise ValueError("budget outside scanned range")


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    prior = json.loads((RESULTS / "phase0_online.json").read_text(encoding="utf-8"))["results"]
    blob = torch.load(OUT / "survey_tables.pt")
    items, tables = blob["items"], blob["tables"]
    model = H.load_model()
    xr, xp, y, groups, *_ = P.fit_rows(model, items, tables, P.FIT)
    rep_model, _ = P.fit_ridge(torch.cat([xr, xp], 1), y, groups, (1e2, 1e3, 1e4, 1e5))
    pos_model, _ = P.fit_ridge(xp, y, groups, (0.01, 0.1, 1.0, 10.0, 100.0))

    def make(name, nu):
        def pol(z, dz, c, prev_len):
            pf = torch.tensor(P.pos_feats(c, prev_len), dtype=torch.float64)
            if name == "position":
                pred = pos_model(pf[None])[0]
            else:
                onehot = torch.tensor([1.0 if dz == d else 0.0 for d in P.V3], dtype=torch.float64)
                pred = rep_model(torch.cat([z.double(), onehot, pf])[None])[0]
            return P.V3[int((pred + nu * P.R_REL).argmin())]
        return pol

    cache = {}

    def test_stats(name, nu):
        if (name, nu) not in cache:
            cache[(name, nu)] = []
            for k in P.TEST:
                nll, depths = P.retry(P.online, model, items[k], make(name, nu))
                cache[(name, nu)].append(P.doc_stats(items[k], nll, depths))
        return cache[(name, nu)]

    uniform = {}
    for k in P.TEST:
        uniform[k] = {}
        for d in P.V3:
            nll, depths = P.retry(P.online, model, items[k], lambda z, dz, c, pl, d=d: d)
            uniform[k][d] = P.doc_stats(items[k], nll, depths)

    out = {}
    for B_key, res in prior["budgets"].items():
        B = float(B_key)
        support = [tuple(s) for s in res["static"]["support"]]
        static_doc = [sum(w * uniform[k][d]["mean_nll"] for d, w in support) for k in P.TEST]
        static_cost = sum(sum(w * uniform[k][d]["cost"] for d, w in support) for k in P.TEST)
        base = sum(uniform[k][16]["base16"] for k in P.TEST)
        entry = {"static": {"doc_mean_nll": sum(static_doc) / len(static_doc), "relative_cost": static_cost / base}}
        docs = {}
        for name in ("position", "rep"):
            lo, hi, w = bracket(prior["cal_scans"][name], B)
            s_lo, s_hi = test_stats(name, lo), test_stats(name, hi)
            doc = [(1 - w) * a["mean_nll"] + w * b["mean_nll"] for a, b in zip(s_lo, s_hi)]
            cost = sum((1 - w) * a["cost"] + w * b["cost"] for a, b in zip(s_lo, s_hi)) / base
            docs[name] = doc
            entry[name] = {"nu_low_cost": lo, "nu_high_cost": hi, "weight_on_high_cost_nu": w,
                           "doc_mean_nll": sum(doc) / len(doc), "relative_cost": cost,
                           "gain_vs_static": P.bootstrap([s - d for s, d in zip(static_doc, doc)])}
        entry["rep_vs_position"] = P.bootstrap([p - r for p, r in zip(docs["position"], docs["rep"])])
        out[B_key] = entry
        print(json.dumps({B_key: entry}), flush=True)
    write_json(RESULTS / "phase0_matched.json", receipt(
        "chunk_depth_v3_1_phase0_matched_budget", note="document-level mixture of two nu values bracketing the "
        "budget on cal; weights fixed on cal and applied on test", results=out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
