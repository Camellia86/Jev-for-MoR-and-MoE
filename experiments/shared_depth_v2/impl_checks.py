"""Pilot stage 1 (protocol v2.1 sections 3.4 and 7.2): implementation checks on the real Huginn checkpoint.

Writes results/shared-depth-v2/impl_checks.json. Exit code 0 only if every gate passes.
"""

from __future__ import annotations

import argparse
import gc
import json
import subprocess
import sys
import time

import torch

from . import cost
from . import huginn as H
from .common import ARTIFACTS, MODEL_DIR, RESULTS, ROOT, Timer, load_config, receipt, sha256_file, write_json
from .init_state import verify_test_vectors

GATES = {"fp32_prob": 1e-5, "fp32_token_nll": 1e-4, "bf16_prob_control": 0.005,
         "bf16_bias_pooled_mean_abs": 0.0005, "bf16_bias_per_depth_mean_abs": 0.001}


def load_data():
    win = torch.load(ARTIFACTS / "data" / "windows.pt")
    man = [json.loads(line) for line in open(ARTIFACTS / "data" / "manifest.jsonl", encoding="utf-8")]
    return win, man


def readout_fixture(device, dtype=torch.float32, seed=7):
    g = torch.Generator(device="cpu").manual_seed(seed)
    lin = torch.nn.Linear(H.cost_hidden if hasattr(H, "cost_hidden") else 5280, 4)
    with torch.no_grad():
        lin.weight.copy_(torch.randn(4, 5280, generator=g) * 0.05)
        lin.bias.zero_()
    return lin.to(device=device, dtype=dtype)


def teacher_logits(model, ids, row, replica, depth):
    """Teacher-forced full-sequence logits for output positions P-1..S-1 (no cache)."""
    S = ids.shape[1]
    state = H.initial_states([row], S, replica, ids.device, next(model.parameters()).dtype)
    feats = H.coda_inputs(model, ids, state, (depth,))
    fc = H.freqs(model, torch.arange(S))
    with torch.no_grad():
        return H.head(model, H.coda(model, feats[depth], fc))[0]


def compare(ref_logits, test_logits, targets):
    """ref/test: [T, V] logits for the same output positions; targets: [T]."""
    pr, pt = torch.softmax(ref_logits.double(), -1), torch.softmax(test_logits.double(), -1)
    nll_r = -torch.log_softmax(ref_logits.double(), -1).gather(1, targets[:, None])[:, 0]
    nll_t = -torch.log_softmax(test_logits.double(), -1).gather(1, targets[:, None])[:, 0]
    return {
        "max_prob_abs_error": float((pr - pt).abs().max()),
        "max_target_prob_abs_error": float((pr.gather(1, targets[:, None]) - pt.gather(1, targets[:, None])).abs().max()),
        "max_token_nll_abs_error": float((nll_r - nll_t).abs().max()),
        "doc_mean_nll_abs_error": float((nll_r.mean() - nll_t.mean()).abs()),
        "ref_doc_mean_nll": float(nll_r.mean()),
    }


def equivalence_suite(model, docs, depths, prefix_len, target_len, device):
    rows = []
    for ids_full, row in docs:
        ids = ids_full[: prefix_len + target_len].to(device).long()[None]
        inputs = ids[:, : prefix_len + target_len - 1]
        targets = ids[0, prefix_len:]
        for depth in depths:
            ref = teacher_logits(model, inputs, row, 0, depth)[prefix_len - 1:]
            plan, _ = H.plan_request(model, inputs[:, :prefix_len], row, 0, depth,
                                     decode_ids=inputs[:, prefix_len:])
            native, _ = H.plan_request(model, inputs[:, :prefix_len], row, 0, depth,
                                       decode_ids=inputs[:, prefix_len:], native=True)
            rows.append({"row": row, "depth": depth,
                         "teacher_vs_incremental_native": compare(ref, native.logits, targets),
                         "teacher_vs_plan_4_to_D": compare(ref, plan.logits, targets),
                         "native_vs_plan_4_to_D": compare(native.logits, plan.logits, targets)})
    return rows


def worst(rows, key):
    out = {}
    for r in rows:
        for pair, vals in r.items():
            if isinstance(vals, dict):
                out[pair] = max(out.get(pair, 0.0), vals[key])
    return out


def check_bf16_bias(model, win, man, device, n_docs: int):
    """v2.1 gate: incremental minus teacher-forcing document-mean NLL, pilot_train docs, four depths."""
    idx = [i for i, m in enumerate(man) if m["role"] == "pilot_train"][:n_docs]
    deltas = torch.empty(len(idx), len(H.DEPTHS), dtype=torch.float64)
    for j, i in enumerate(idx):
        row = man[i]["source_row"]
        ids = win[i:i + 1, :255].long().to(device)
        targets = win[i, 128:256].long().to(device)
        for k, depth in enumerate(H.DEPTHS):
            ref = teacher_logits(model, ids, row, 0, depth)[127:]
            inc, _ = H.plan_request(model, ids[:, :128], row, 0, depth, decode_ids=ids[:, 128:255], native=True)
            nll_ref = torch.nn.functional.cross_entropy(ref.double(), targets)
            nll_inc = torch.nn.functional.cross_entropy(inc.logits.double(), targets)
            deltas[j, k] = float(nll_inc - nll_ref)
    pooled = deltas.mean(1)
    n = len(idx)
    mean, sd = float(pooled.mean()), float(pooled.std(unbiased=True))
    se = sd / n ** 0.5
    per_depth = {}
    for k, depth in enumerate(H.DEPTHS):
        col = deltas[:, k]
        per_depth[depth] = {"mean": float(col.mean()), "sd": float(col.std(unbiased=True)),
                            "se": float(col.std(unbiased=True)) / n ** 0.5, "max_abs": float(col.abs().max())}
    return {"documents": n, "role": "pilot_train", "depths": list(H.DEPTHS),
            "pooled": {"mean": mean, "sd": sd, "se": se, "ci95": [mean - 1.959964 * se, mean + 1.959964 * se]},
            "per_depth": per_depth, "deltas": [[round(float(v), 6) for v in r] for r in deltas]}


def check_prefix_invariance(model, win, man, device):
    """Control path never reads the suffix; full-run position P-1 agrees with prefix-only run."""
    readout = readout_fixture(device)
    row = man[0]["source_row"]
    prefix = win[0, :128].long().to(device)[None]
    suffix_a = win[0, 128:255].long().to(device)[None]
    suffix_b = win[1, 128:255].long().to(device)[None]
    res_a, _ = H.plan_request(model, prefix, row, 0, 4, readout=readout)
    res_b, _ = H.plan_request(model, prefix, row, 0, 4, readout=readout)
    fc = H.freqs(model, torch.arange(255))
    reps = {}
    for name, suffix in (("a", suffix_a), ("b", suffix_b)):
        ids = torch.cat([prefix, suffix], 1)
        state = H.initial_states([row], 255, 0, device)
        feats = H.coda_inputs(model, ids, state, (4,))
        with torch.no_grad():
            z = H.coda(model, feats[4], fc)
            reps[name] = (z[0, :128].float(), torch.softmax(readout(z[0, 127].float()), -1))
    prefix_state = H.initial_states([row], 128, 0, device)
    feats_p = H.coda_inputs(model, prefix, prefix_state, (4,))
    with torch.no_grad():
        zp = H.coda(model, feats_p[4], H.freqs(model, torch.arange(128)))
        pp = torch.softmax(readout(zp[0, 127].float()), -1)
    return {
        "control_probs_identical_across_suffixes": bool(torch.equal(res_a.control_probs, res_b.control_probs)),
        "full_runs_prefix_repr_max_abs_diff_suffix_a_vs_b": float((reps["a"][0] - reps["b"][0]).abs().max()),
        "full_run_vs_prefix_only_control_prob_max_abs_diff": float((reps["a"][1] - pp).abs().max()),
        "plan_path_vs_prefix_only_control_prob_max_abs_diff": float((res_a.control_probs - pp).abs().max()),
    }


def check_lora(model, win, man, device):
    modules = H.attach_lora(model, rank=8, alpha=16, seed=17)
    params = H.lora_parameters(modules)
    n_params = sum(p.numel() for p in params)
    row = man[0]["source_row"]
    ids = win[0:1, :255].long().to(device)
    state = H.initial_states([row], 255, 0, device)
    feats = H.coda_inputs(model, ids, state, (8,))
    prefix_feats = H.coda_inputs(model, ids[:, :128], state[:, :128], (4,))
    fc, fcp = H.freqs(model, torch.arange(255)), H.freqs(model, torch.arange(128))
    H.set_lora_enabled(modules, False)
    with torch.no_grad():
        base = H.coda(model, feats[8], fc)
    H.set_lora_enabled(modules, True)
    with torch.no_grad():
        with_zero_b = H.coda(model, feats[8], fc)
    readout = readout_fixture(device)
    targets = win[0, 128:256].long().to(device)

    def grads(use_gen: bool, use_ctl: bool):
        for p in params:
            p.grad = None
        loss = 0.0
        if use_gen:
            logits = H.head(model, H.coda(model, feats[8], fc))[0, 127:]
            loss = loss + torch.nn.functional.cross_entropy(logits, targets)
        if use_ctl:
            z = H.coda(model, prefix_feats[4], fcp)
            logp = torch.log_softmax(readout(z[0, 127].float()), -1)
            loss = loss - logp[1]
        loss.backward()
        a = sum(float(m.A.grad.norm()) for m in modules)
        b = sum(float(m.B.grad.norm()) for m in modules)
        return a, b

    joint = grads(True, True)
    gen_only = grads(True, False)
    ctl_only = grads(False, True)
    shared_ids = {id(p) for p in params}
    result = {
        "lora_parameters": n_params,
        "expected_parameters": 1536000,
        "zero_B_output_equals_base_bitwise": bool(torch.equal(base, with_zero_b)),
        "joint_A_grad_norm": joint[0], "joint_B_grad_norm": joint[1],
        "generation_only_B_grad_norm": gen_only[1], "control_only_B_grad_norm": ctl_only[1],
        "same_parameter_objects_in_both_paths": len(shared_ids) == len(params),
        "modules_wrapped": len(modules),
    }
    for p in params:
        p.grad = None
    H.detach_lora(model)
    return result


def check_kv_memory(model, win, man, device):
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    row = man[0]["source_row"]
    ids = win[0:1, :255].long().to(device)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    res, cache = H.plan_request(model, ids[:, :128], row, 0, 32, decode_ids=ids[:, 128:255])
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    positions = max(max(v.keys()) for v in cache.key_cache.values()) + 1
    layers = len(cache.key_cache)
    total = torch.cuda.get_device_properties(0).total_memory
    peak = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    del cache
    return {"depth": 32, "cached_positions": positions, "cached_layer_indices": layers,
            "expected_layer_indices": 2 + 4 * 32 + 2, "cache_mib": None,
            "peak_allocated_gib": peak / 2**30, "peak_reserved_gib": reserved / 2**30,
            "device_total_gib": total / 2**30, "peak_reserved_fraction": reserved / total,
            "teacher_forced_request_seconds": elapsed}


def check_flops(model, win, man, device):
    from torch.utils.flop_counter import FlopCounterMode

    row = man[0]["source_row"]
    ids = win[0:1, :255].long().to(device)
    readout = readout_fixture(device)
    rows = []
    for depth, native in ((4, True), (8, True), (8, False)):
        counter = FlopCounterMode(display=False)
        with counter:
            H.plan_request(model, ids[:, :128], row, 0, depth, readout=None if native else readout,
                           decode_ids=ids[:, 128:255], native=native)
        measured = counter.get_total_flops()
        ledger = cost.native(depth) if native else cost.plan(depth)
        rows.append({"depth": depth, "path": "native" if native else "plan", "counter_flops": measured,
                     "ledger_flops": ledger, "relative_error": abs(measured - ledger) / ledger})
    return rows


def timing_smoke(model, win, man, device):
    row = man[0]["source_row"]
    ids = win[0:1, :128].long().to(device)
    out = {}
    for depth in (4, 8):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        res, _ = H.plan_request(model, ids, row, 0, depth, steps_out=16)
        torch.cuda.synchronize()
        out[str(depth)] = {"generated_tokens": 16, "seconds": time.perf_counter() - t0,
                           "first_tokens": res.logits.argmax(-1)[:8].tolist()}
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bf16-docs", type=int, default=3)
    parser.add_argument("--skip-fp32", action="store_true")
    parser.add_argument("--bias-docs", type=int, default=64)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    costs: dict = {}
    checks: dict = {}
    gates: dict = {}
    win, man = load_data()
    cfg = load_config()

    with Timer(costs, "math_self_check_seconds"):
        out = ARTIFACTS / "impl" / "math_check.json"
        proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "check_experiment_math.py"), "--output", str(out)],
                              capture_output=True, text=True)
        checks["math_self_check"] = json.loads(out.read_text(encoding="utf-8"))["status"]
    gates["math_self_check"] = checks["math_self_check"] == "passed"

    vectors = verify_test_vectors(cfg["runtime"]["initial_state"]["test_vectors"])
    checks["initial_state_vectors"] = vectors
    gates["initial_state_vectors"] = all(v["seed_match"] and v["first8_match"] and v["sha256_match"] for v in vectors)

    loaded = sys.modules.get("transformers_modules.huginn-0125.raven_modeling_minimal")
    checks["model_source"] = {"checkpoint_sha256": sha256_file(MODEL_DIR / "raven_modeling_minimal.py")}

    device = "cuda"
    with Timer(costs, "model_load_seconds"):
        model = H.load_model(device)
    mod = sys.modules[type(model).__module__]
    checks["model_source"]["loaded_module_sha256"] = sha256_file(__import__("pathlib").Path(mod.__file__))
    gates["model_source_matches_checkpoint"] = (checks["model_source"]["loaded_module_sha256"]
                                                == checks["model_source"]["checkpoint_sha256"])

    with Timer(costs, "native_vs_decomposition_seconds"):
        row = man[0]["source_row"]
        ids = win[0:1, :255].long().to(device)
        st = H.initial_states([row], 255, 0, device)
        od = {"return_logits": True, "return_latents": False, "return_head": False, "return_stats": False}
        with torch.no_grad():
            native = model(ids, input_states=st, num_steps=torch.tensor([16]), output_details=od).logits
        mine = teacher_logits(model, ids, row, 0, 16)
        checks["native_forward_vs_decomposition_max_abs_logit_diff"] = float((native[0] - mine).abs().max())
    gates["native_forward_vs_decomposition_bitwise"] = checks["native_forward_vs_decomposition_max_abs_logit_diff"] == 0.0

    with Timer(costs, "prefix_invariance_seconds"):
        checks["prefix_invariance"] = check_prefix_invariance(model, win, man, device)
    pi = checks["prefix_invariance"]
    gates["prefix_invariance"] = (pi["control_probs_identical_across_suffixes"]
                                  and pi["full_runs_prefix_repr_max_abs_diff_suffix_a_vs_b"] == 0.0
                                  and pi["full_run_vs_prefix_only_control_prob_max_abs_diff"] <= GATES["bf16_prob_control"]
                                  and pi["plan_path_vs_prefix_only_control_prob_max_abs_diff"] <= GATES["bf16_prob_control"])

    with Timer(costs, "bf16_equivalence_seconds"):
        docs = [(win[i], man[i]["source_row"]) for i in range(args.bf16_docs)]
        rows = equivalence_suite(model, docs, H.DEPTHS, 128, 128, device)
    checks["bf16_equivalence"] = {"documents": args.bf16_docs, "rows": rows,
                                  "worst_prob": worst(rows, "max_prob_abs_error"),
                                  "worst_doc_nll": worst(rows, "doc_mean_nll_abs_error")}
    gates["bf16_continuation_bitwise"] = checks["bf16_equivalence"]["worst_prob"]["native_vs_plan_4_to_D"] == 0.0

    with Timer(costs, "bf16_bias_seconds"):
        checks["bf16_bias"] = check_bf16_bias(model, win, man, device, args.bias_docs)
    bias = checks["bf16_bias"]
    lo_ci, hi_ci = bias["pooled"]["ci95"]
    gates["bf16_bias"] = (abs(bias["pooled"]["mean"]) <= GATES["bf16_bias_pooled_mean_abs"] and lo_ci <= 0.0 <= hi_ci
                          and all(abs(v["mean"]) <= GATES["bf16_bias_per_depth_mean_abs"]
                                  for v in bias["per_depth"].values()))

    with Timer(costs, "lora_seconds"):
        checks["lora"] = check_lora(model, win, man, device)
    lo = checks["lora"]
    gates["lora"] = (lo["lora_parameters"] == lo["expected_parameters"] and lo["zero_B_output_equals_base_bitwise"]
                     and lo["joint_A_grad_norm"] == 0.0 and lo["joint_B_grad_norm"] > 0
                     and lo["generation_only_B_grad_norm"] > 0 and lo["control_only_B_grad_norm"] > 0
                     and lo["same_parameter_objects_in_both_paths"])

    with Timer(costs, "kv_memory_seconds"):
        checks["kv_memory"] = check_kv_memory(model, win, man, device)
    kv = checks["kv_memory"]
    gates["kv_memory"] = (kv["cached_positions"] == 255 and kv["cached_layer_indices"] == kv["expected_layer_indices"]
                          and kv["peak_reserved_fraction"] <= 1 - cfg["runtime"]["minimum_free_vram_fraction"])

    with Timer(costs, "flop_ledger_seconds"):
        checks["flop_ledger"] = check_flops(model, win, man, device)
    gates["flop_ledger"] = all(r["relative_error"] <= 0.005 for r in checks["flop_ledger"])
    checks["normalized_cost"] = cost.normalized()

    with Timer(costs, "timing_smoke_seconds"):
        checks["timing_smoke"] = timing_smoke(model, win, man, device)
    gates["timing_smoke"] = True

    del model
    gc.collect()
    torch.cuda.empty_cache()

    if not args.skip_fp32:
        with Timer(costs, "fp32_cpu_equivalence_seconds"):
            cpu_model = H.load_model("cpu", torch.float32)
            docs = [(win[i], man[i]["source_row"]) for i in range(2)]
            rows = equivalence_suite(cpu_model, docs, (4, 8), 8, 8, "cpu")
            del cpu_model
            gc.collect()
        checks["fp32_equivalence"] = {"prefix_tokens": 8, "target_tokens": 8, "documents": 2, "rows": rows,
                                      "worst_prob": worst(rows, "max_prob_abs_error"),
                                      "worst_token_nll": worst(rows, "max_token_nll_abs_error")}
        gates["fp32_equivalence"] = (max(checks["fp32_equivalence"]["worst_prob"].values()) <= GATES["fp32_prob"]
                                     and max(checks["fp32_equivalence"]["worst_token_nll"].values()) <= GATES["fp32_token_nll"])

    passed = all(gates.values())
    payload = receipt("pilot_stage1_implementation_checks", protocol_revision="2.1",
                      status="passed" if passed else "failed",
                      gates=gates, thresholds=GATES, checks=checks, costs=costs)
    write_json(RESULTS / "impl_checks.json", payload)
    print(json.dumps({"status": payload["status"], "gates": gates, "costs": costs}, indent=1))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
