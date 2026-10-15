"""Pilot step 1 (plan v3.1 section 7.2): implementation checks. Writes results/chunk-depth-v31/checks.json."""

from __future__ import annotations

import hashlib
import json
import struct

import torch

from experiments.shared_depth_v2 import cost as C
from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import ROOT, receipt, write_json
from experiments.chunk_depth_v31.data import OUT, RESULTS, state_matrix
from experiments.chunk_depth_v31 import run as R


def check_vectors():
    cfg = json.loads((ROOT / "docs" / "EXPERIMENT_CONFIG.v3.1.json").read_text(encoding="utf-8"))
    rows = []
    for v in cfg["data"]["initial_state"]["test_vectors"]:
        x = state_matrix(v["split"], v["row"], v["position"] + 1, v["replica"])[v["position"]]
        raw = b"".join(struct.pack("<h", t) for t in x.view(torch.int16).tolist())
        rows.append(raw[:16].hex() == v["first_8_bf16_le_hex"] and hashlib.sha256(raw).hexdigest() == v["sha256_bf16_le"])
    return {"vectors": len(rows), "all_match": all(rows)}


@torch.no_grad()
def check_native(model, item):
    ids = torch.tensor(item["ids"], device="cuda")[None]
    S = ids.shape[1]
    st = R.item_state(item, S)
    od = {"return_logits": True, "return_latents": False, "return_head": False, "return_stats": False}
    native = model(ids, input_states=st, num_steps=torch.tensor([8]), output_details=od).logits[0]
    fc = H.freqs(model, torch.arange(S))
    mine = H.head(model, H.coda(model, H.coda_inputs(model, ids, st, (8,))[8], fc))[0]
    return {"max_abs_logit_diff": float((native - mine).abs().max())}


def check_chunk_bias(model, items, feats, n_docs=64, depths=(4, 16)):
    deltas = torch.empty(n_docs, len(depths), dtype=torch.float64)
    for j in range(n_docs):
        item, f = items[j], feats[j]
        _, lines, _ = R.lines_of(item)
        sc = [t for (a, b) in lines for t in R.scored_in(item, a, b)]
        for k, d in enumerate(depths):
            nll, _ = R.retry(R.online, model, item, lambda z, dz, c, pl, h, d=d: d, q_depth=d)
            table = f["nll"][R.V3.index(d), sc].double().mean()
            deltas[j, k] = float(nll[sc].double().mean() - table)
    pooled = deltas.mean(1)
    se = float(pooled.std()) / n_docs ** 0.5
    mean = float(pooled.mean())
    per_depth = {d: {"mean": float(deltas[:, k].mean()), "sd": float(deltas[:, k].std())} for k, d in enumerate(depths)}
    passed = abs(mean) <= 0.0005 and mean - 1.96 * se <= 0 <= mean + 1.96 * se and \
        all(abs(v["mean"]) <= 0.001 for v in per_depth.values())
    return {"documents": n_docs, "pooled_mean": mean, "pooled_se": se, "per_depth": per_depth, "passed": passed}


@torch.no_grad()
def check_decision_rep(model, items):
    """The tensor handed to the policy for line c must be the coda output of the previous segment's last
    position, i.e. the same row whose head output predicts line c's first token. Identity is checked by
    capturing the segment outputs; the head recomputed on that single row is compared numerically
    (a single-row GEMM is not bitwise identical to the batched one in BF16)."""
    captured = []
    original = R.run_segment

    def spy(model_, cache, ids, state, a, b, d):
        z, logits = original(model_, cache, ids, state, a, b, d)
        captured.append(z[-1].float().cpu())
        return z, logits

    identical, max_diff, lines_total = True, 0.0, 0
    R.run_segment = spy
    try:
        for item in items:
            captured.clear()
            nll, depths, reps = R.online(model, item, lambda z, dz, c, pl, h: R.V3[c % 5], return_reps=True)
            _, lines, _ = R.lines_of(item)
            for c, (a, b) in enumerate(lines):
                identical &= bool(torch.equal(reps[c], captured[c]))
                z = reps[c].to("cuda", dtype=torch.bfloat16)
                logits = H.head(model, z[None, None])[0, 0].float()
                ce = torch.nn.functional.cross_entropy(logits[None], torch.tensor([item["ids"][a]], device="cuda"))
                max_diff = max(max_diff, abs(float(ce) - float(nll[a - 1])))
                lines_total += 1
    finally:
        R.run_segment = original
    return {"lines": lines_total, "identical_tensor": identical, "max_abs_nll_diff_single_row_head": max_diff,
            "passed": identical and max_diff <= 0.02}


def check_latest_m4():
    """Toy cache: positions 0-2 computed to depth 1 (steps 2-5), position 3 to depth 2 (steps 2-9).
    Updating step 6 for position 3 must return step-2 entries for positions 0-2."""
    cache = R.new_cache()
    hd = 4
    val = lambda tag, n: torch.full((1, 1, n, hd), float(tag))
    cache.update(val(0, 3), val(0, 3), torch.tensor(0))
    for s in range(1, 6):
        cache.update(val(s, 3), val(s, 3), torch.tensor(s))
    cache.update(val(100, 1), val(100, 1), torch.tensor(0))
    for s in range(1, 6):
        cache.update(val(100 + s, 1), val(100 + s, 1), torch.tensor(s))
    k, _ = cache.update(val(106, 1), val(106, 1), torch.tensor(6))
    got = k[0, 0, :, 0].tolist()
    return {"returned_step_tags": got, "expected": [2.0, 2.0, 2.0, 106.0], "passed": got == [2.0, 2.0, 2.0, 106.0]}


def check_lora(model, item):
    modules = H.attach_lora(model, 8, 16, seed=17)
    params = H.lora_parameters(modules)
    ids = torch.tensor(item["ids"], device="cuda")[None]
    S = ids.shape[1]
    st = R.item_state(item, S)
    feats = H.coda_inputs(model, ids, st, (8,))
    fc = H.freqs(model, torch.arange(S))
    H.set_lora_enabled(modules, False)
    with torch.no_grad():
        base = H.coda(model, feats[8], fc)
    H.set_lora_enabled(modules, True)
    with torch.no_grad():
        zero_b = H.coda(model, feats[8], fc)
    readout = torch.nn.Linear(5280, 5).cuda()
    q_end, _, _ = R.lines_of(item)

    def norms(gen, ctl):
        for p in params:
            p.grad = None
        z = H.coda(model, feats[8], fc)
        loss = 0.0
        if gen:
            loss = loss + torch.nn.functional.cross_entropy(H.head(model, z)[0, :-1], ids[0, 1:])
        if ctl:
            loss = loss - torch.log_softmax(readout(z[0, q_end - 1].float()), -1)[2]
        loss.backward()
        return sum(float(m.A.grad.norm()) for m in modules), sum(float(m.B.grad.norm()) for m in modules)

    joint, gen, ctl = norms(True, True), norms(True, False), norms(False, True)
    H.detach_lora(model)
    return {"parameters": sum(p.numel() for p in params), "zero_B_equals_base": bool(torch.equal(base, zero_b)),
            "joint_A_grad": joint[0], "joint_B_grad": joint[1], "gen_B_grad": gen[1], "ctl_B_grad": ctl[1],
            "passed": sum(p.numel() for p in params) == 1536000 and bool(torch.equal(base, zero_b))
            and joint[0] == 0.0 and joint[1] > 0 and gen[1] > 0 and ctl[1] > 0}


@torch.no_grad()
def check_ledger(model, item, d=4):
    """Token-by-token execution at depth d versus the per-position ledger sum."""
    from torch.utils.flop_counter import FlopCounterMode
    ids = torch.tensor(item["ids"], device="cuda")[None]
    S = ids.shape[1]
    st = R.item_state(item, S)
    cache = R.new_cache()
    counter = FlopCounterMode(display=False)
    with counter:
        for t in range(S):
            R.run_segment(model, cache, ids, st, t, t + 1, d)
    measured = counter.get_total_flops()
    ledger = sum(C.stack(1, t + 1, d) + C.lm_head(1) for t in range(S))
    return {"measured": measured, "ledger": ledger, "relative_error": abs(measured - ledger) / ledger,
            "passed": abs(measured - ledger) / ledger <= 0.005}


def main() -> int:
    torch.backends.cuda.matmul.allow_tf32 = False
    items = torch.load(OUT / "items_pilot_train.pt")
    feats = torch.load(OUT / "features_pilot_train.pt")
    model = H.load_model()
    out = {"initial_state_vectors": check_vectors()}
    out["native_vs_decomposition"] = check_native(model, items[0])
    out["latest_m4_toy"] = check_latest_m4()
    out["decision_representation"] = check_decision_rep(model, items[1:9])
    out["lora"] = check_lora(model, items[2])
    out["ledger"] = check_ledger(model, items[3])
    print(json.dumps(out, indent=1), flush=True)
    out["chunk_execution_bias"] = check_chunk_bias(model, items, feats)
    gates = {"initial_state_vectors": out["initial_state_vectors"]["all_match"],
             "native_vs_decomposition": out["native_vs_decomposition"]["max_abs_logit_diff"] == 0.0,
             "latest_m4_toy": out["latest_m4_toy"]["passed"],
             "decision_representation": out["decision_representation"]["passed"],
             "lora": out["lora"]["passed"], "ledger": out["ledger"]["passed"],
             "chunk_execution_bias": out["chunk_execution_bias"]["passed"]}
    status = "passed" if all(gates.values()) else "failed"
    write_json(RESULTS / "checks.json", receipt("chunk_depth_v3_1_pilot_checks", status=status, gates=gates, checks=out))
    print(json.dumps({"status": status, "gates": gates, "bias": out["chunk_execution_bias"]}, indent=1))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
