"""Phase 0 addendum: question-level decision (original idea) at exactly matched budget.

One depth per problem, chosen from the representation at the end of the question (question at 16) by
argmin predicted per-token NLL + nu * r(d). Realised quality equals the uniform-line run at the chosen
depth. A fine nu grid is scanned on cal; the two nu values bracketing the budget are mixed at document
level so the cal cost equals the budget; the same pair and weight are applied on test.
"""

from __future__ import annotations

import json

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, write_json
from experiments.chunk_depth_v3 import phase0_online as P
from experiments.chunk_depth_v3.survey import OUT, RESULTS, state_matrix


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    blob = torch.load(OUT / "survey_tables.pt")
    items, tables = blob["items"], blob["tables"]
    model = H.load_model()
    *_, qx, qy, qg = P.fit_rows(model, items, tables, P.FIT)
    q_model, q_alpha = P.fit_ridge(qx, qy, qg, (1e2, 1e3, 1e4, 1e5))
    idx = list(P.CAL) + list(P.TEST)
    uniform, q_pred = {}, {}
    for k in idx:
        uniform[k] = {}
        for d in P.V3:
            nll, depths = P.retry(P.online, model, items[k], lambda z, dz, c, pl, d=d: d)
            uniform[k][d] = P.doc_stats(items[k], nll, depths)
        q_end, _, _ = P.lines_of(items[k])
        ids = torch.tensor(items[k]["ids"][:q_end], device="cuda")[None]
        state = state_matrix(items[k]["row"], q_end).cuda()[None]
        z, _ = P.run_segment(model, P.new_cache(), ids, state, 0, q_end, 16)
        q_pred[k] = q_model(z[-1].double().cpu()[None])[0]
    torch.save({"uniform": uniform, "q_pred": q_pred}, OUT / "phase0_uniform_cal_test.pt")

    def evaluate(keys, nu):
        pick = {k: P.V3[int((q_pred[k] + nu * P.R_REL).argmin())] for k in keys}
        cost = sum(uniform[k][pick[k]]["cost"] for k in keys) / sum(uniform[k][16]["base16"] for k in keys)
        return pick, cost

    grid = [i * 0.05 for i in range(0, 241)]
    out = {"alpha": q_alpha}
    for B in (0.5, 0.6):
        scan = [(nu, evaluate(P.CAL, nu)[1]) for nu in grid]
        pair = None
        for (nu_a, c_a), (nu_b, c_b) in zip(scan, scan[1:]):
            if c_b <= B <= c_a and c_a > c_b:
                pair = (nu_a, nu_b, (B - c_b) / (c_a - c_b))
                break
        support = P.best_static(uniform, P.CAL, B)
        static_doc = {k: sum(w * uniform[k][d]["mean_nll"] for d, w in support) for k in P.TEST}
        nu_hi_cost, nu_lo_cost, w = pair
        pick_hi, _ = evaluate(P.TEST, nu_hi_cost)
        pick_lo, _ = evaluate(P.TEST, nu_lo_cost)
        doc = {k: w * uniform[k][pick_hi[k]]["mean_nll"] + (1 - w) * uniform[k][pick_lo[k]]["mean_nll"]
               for k in P.TEST}
        cost = sum(w * uniform[k][pick_hi[k]]["cost"] + (1 - w) * uniform[k][pick_lo[k]]["cost"] for k in P.TEST)
        base = sum(uniform[k][16]["base16"] for k in P.TEST)
        out[str(B)] = {"nu_high_cost": nu_hi_cost, "nu_low_cost": nu_lo_cost, "weight_on_high_cost_nu": w,
                       "doc_mean_nll": sum(doc.values()) / len(doc), "relative_cost": cost / base,
                       "static_doc_mean_nll": sum(static_doc.values()) / len(static_doc),
                       "gain_vs_static": P.bootstrap([static_doc[k] - doc[k] for k in P.TEST])}
        print(json.dumps({B: out[str(B)]}), flush=True)
    write_json(RESULTS / "phase0_question.json", receipt("chunk_depth_v3_1_phase0_question_level", results=out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
