"""CPU-only audit of the completed frozen-theta_S readout; never rerun L-BFGS.

Load the existing ~0.78 GB tables once and build rows in small document batches.
The full [57071, 5290] FP64 design matrix is never allocated. No reserved
stage-two items are opened. Run this audit serially before mechanism training.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch

from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31.train import doc_structure, load_kv
from experiments.chunk_depth_v32.common import OUT as OUT32, RESULTS as RESULTS32, load_role
from experiments.chunk_depth_v32.head import build_rows, soft_labels
from . import common as C


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(actual, expected, label, atol=1e-10, rtol=1e-10):
    require(math.isclose(float(actual), float(expected), abs_tol=atol, rel_tol=rtol),
            f"{label}: {actual} != {expected}")
    return {"actual": float(actual), "expected": float(expected),
            "absolute_difference": abs(float(actual) - float(expected))}


def row_batches(items, tables, batch_documents, kappa_s, kappa_e):
    for begin in range(0, len(items), batch_documents):
        end = min(begin + batch_documents, len(items))
        yield begin, end, build_rows(items[begin:end], tables[begin:end], kappa_s, kappa_e)


def audit_selection(blob, receipt, stage1):
    expected = {"gamma": 0.01, "lambda": 1.0, "tau": 0.05}
    for key, value in expected.items():
        close(blob[key], value, f"checkpoint {key}", atol=0, rtol=0)
        close(C.CONFIG["head"][key], value, f"follow-up config {key}", atol=0, rtol=0)
    require(C.CONFIG["head"]["frozen"] is True, "Follow-up head must be frozen")
    close(receipt["tau"], blob["tau"], "head receipt tau")
    close(stage1["lambda"], blob["lambda"], "stage-one lambda")
    close(stage1["gamma"], blob["gamma"], "stage-one gamma")
    by_gamma = receipt["cv"]["by_gamma"]
    gamma = min((float(g) for g in by_gamma), key=lambda g: (by_gamma[str(g)]["held_out_ce"], g))
    close(gamma, blob["gamma"], "CV-selected gamma")
    close(receipt["cv"]["selected_gamma"], gamma, "CV receipt gamma")
    rows = receipt["lambda_rule"]["rows"]
    for row in rows:
        close(row["abs_log_ratio"], abs(math.log(row["greedy_relative_cost"] / 0.5)), "lambda score")
    chosen = min(rows, key=lambda row: (round(row["abs_log_ratio"], 12), row["lambda"]))
    close(chosen["lambda"], blob["lambda"], "lambda rule")
    close(receipt["lambda_rule"]["selected_lambda"], blob["lambda"], "lambda receipt")
    for key, shape in (("W", (5, 5290)), ("beta", (5,)), ("m", (5280,)), ("sd", (5280,))):
        value = blob[key]
        require(tuple(value.shape) == shape and value.dtype == torch.float64 and value.device.type == "cpu",
                f"Invalid head tensor {key}")
        require(bool(torch.isfinite(value).all()) and not value.requires_grad, f"Nonfinite/trainable saved {key}")
    require(bool((blob["sd"] > 0).all()), "Head standard deviations must be positive")
    for key in ("W", "beta"):
        require(torch.equal(blob[key], blob["all_lambdas"][blob["lambda"]][key]),
                f"Selected {key} differs from all_lambdas")
    return {**expected, "selection_reproduced": True, "selected_fit": chosen}


def audit_normalization(items, tables, blob, receipt, batch_documents, guard):
    kappa_s, kappa_e = load_kv()
    count, scored_lines, total_weight = 0, 0, 0.0
    mean = torch.zeros(5280, dtype=torch.float64)
    m2 = torch.zeros_like(mean)
    max_document_weight_error = 0.0
    for begin, end, rows in row_batches(items, tables, batch_documents, kappa_s, kappa_e):
        guard()
        z = rows["z"].double()
        n = len(z)
        part_mean = z.mean(0)
        part_m2 = ((z - part_mean) ** 2).sum(0)
        delta = part_mean - mean
        new_count = count + n
        m2 += part_m2 + delta.square() * (count * n / new_count)
        mean += delta * (n / new_count)
        count = new_count
        weight = torch.zeros(end - begin, dtype=torch.float64).index_add_(0, rows["doc"], rows["w"])
        max_document_weight_error = max(max_document_weight_error, float((weight - 1).abs().max()))
        total_weight += float(rows["w"].sum())
        scored_lines += sum(sum(l["n"] > 0 for l in doc_structure(item)) for item in items[begin:end])
        if end % 256 == 0 or end == len(items):
            print(f"head audit normalization {end}/{len(items)}, rows={count}", flush=True)
    sd = torch.sqrt(m2 / (count - 1)) + 1e-6
    require(count == receipt["rows"] == 57071, "Unexpected readout row count")
    require(len(items) == receipt["documents"] == 4111, "Unexpected readout document count")
    require(scored_lines == 14703, "Unexpected number of scored solution lines")
    require(max_document_weight_error < 1e-12, "Per-document readout weights do not sum to one")
    close(total_weight, len(items), "total row weight", atol=1e-9)
    require(torch.allclose(mean, blob["m"], atol=1e-10, rtol=1e-10), "Stored mean differs elementwise")
    require(torch.allclose(sd, blob["sd"], atol=1e-10, rtol=1e-10), "Stored standard deviation differs elementwise")
    return {"documents": len(items), "rows": count, "scored_lines": scored_lines,
            "weight_sum": total_weight, "max_document_weight_error": max_document_weight_error,
            "mean_max_absolute_difference": float((mean - blob["m"]).abs().max()),
            "sd_max_absolute_difference": float((sd - blob["sd"]).abs().max()),
            "elementwise_mean_sd_passed": True, "mean_sd_atol": 1e-10, "mean_sd_rtol": 1e-10}


def audit_objective(items, tables, blob, receipt, weight_sum, batch_documents, guard):
    kappa_s, kappa_e = load_kv()
    W, beta = [blob[key].clone().requires_grad_(True) for key in ("W", "beta")]
    objective = 0.0
    for begin, end, rows in row_batches(items, tables, batch_documents, kappa_s, kappa_e):
        guard()
        x = torch.cat([(rows["z"].double() - blob["m"]) / blob["sd"], rows["phi"]], 1)
        labels, _ = soft_labels(rows, blob["lambda"])
        logp = torch.log_softmax(x @ W.T + beta, -1)
        loss = -((rows["w"] / weight_sum) * (labels * logp).sum(-1)).sum()
        objective += float(loss.detach())
        loss.backward()
        del x, labels, logp, loss, rows
        if end % 256 == 0 or end == len(items):
            print(f"head audit objective and gradient {end}/{len(items)}", flush=True)
    penalty = blob["gamma"] * W.square().sum()
    objective += float(penalty.detach())
    penalty.backward()
    grad_inf = max(float(W.grad.abs().max()), float(beta.grad.abs().max()))
    expected = receipt["selected_head_recheck"]
    obj_check = close(objective, expected["objective"], "selected head objective", atol=1e-10)
    grad_check = close(grad_inf, expected["grad_inf"], "selected head gradient infinity norm", atol=1e-8, rtol=0)
    require(grad_inf < 1e-5, "Selected readout fails the recorded convergence criterion")
    return {"objective": obj_check, "gradient_infinity_norm": grad_check,
            "converged": True, "gradient_threshold": 1e-5,
            "formula": "sum_rows w/sum(w) * CE(soft_labels, p) + gamma * ||W||_F^2"}


@torch.no_grad()
def audit_greedy_cost(items, tables, blob, expected, guard):
    num = den = 0.0
    for j, (item, table) in enumerate(zip(items, tables)):
        guard()
        previous = R.V3.index(16)
        for i, line in enumerate(doc_structure(item)):
            z = table["reps"][previous, i].double()
            x = torch.cat([(z - blob["m"]) / blob["sd"], torch.tensor(line["phi"], dtype=torch.float64)])
            choice = int((blob["W"] @ x + blob["beta"]).argmax())
            num += float(line["f"][choice])
            den += float(line["f"][R.V3.index(16)])
            previous = choice
        if (j + 1) % 512 == 0 or j + 1 == len(items):
            print(f"head audit greedy cost {j + 1}/{len(items)}", flush=True)
    return close(num / den, expected, "selected head sequential greedy cost", atol=1e-12)


def audit_frozen_head(blob, table, item):
    # Exercise the actual follow-up training adapter on CPU. The only new leaf
    # requiring a gradient is z; no model is loaded and no optimizer is created.
    from .train import FrozenHead
    head = FrozenHead(blob, device="cpu")
    before = {key: getattr(head, key).clone() for key in ("W", "beta", "m", "sd")}
    require(all(not getattr(head, key).requires_grad for key in before), "Follow-up readout is trainable")
    z = table["reps"][4, 0].double().detach().requires_grad_(True)
    logp = head.logp(z, doc_structure(item)[0]["phi"])
    (-logp[0]).backward()
    require(z.grad is not None and bool(torch.isfinite(z.grad).all()) and float(z.grad.norm()) > 0,
            "Frozen head blocks the gradient to its input")
    require(all(getattr(head, key).grad is None and torch.equal(getattr(head, key), before[key]) for key in before),
            "Follow-up readout parameters received a gradient or changed")
    return {"adapter": "train.FrozenHead(device='cpu')", "head_requires_grad": False,
            "head_gradients_are_none": True, "head_tensors_unchanged": True,
            "input_gradient_norm": float(z.grad.norm()), "input_gradient_preserved": True}


def audit_stage1(stage1):
    bracket_checks = {}
    for name in ("head", "R"):
        cal = stage1["calibration"][name]
        br = cal["bracket"]
        points = {point["param"]: point for point in cal["online_points"]}
        hi, lo = points[br["param_high_cost"]], points[br["param_low_cost"]]
        weight = br["weight_high"]
        require(0 <= weight <= 1 and lo["relative_cost"] <= 0.5 <= hi["relative_cost"],
                f"Invalid original {name} online budget bracket")
        mixed_cost = weight * hi["relative_cost"] + (1 - weight) * lo["relative_cost"]
        close(mixed_cost, 0.5, f"original {name} calibration mixture cost", atol=1e-12)
        close(weight * hi["doc_mean_nll"] + (1 - weight) * lo["doc_mean_nll"],
              cal["cal_matched_nll"], f"original {name} calibration mixture NLL")
        bracket_checks[name] = {**br, "matched_calibration_cost": mixed_cost}
    screen = stage1["screen"]
    ratio = screen["head"]["relative_cost"] / screen["R"]["relative_cost"]
    ratio_check = close(ratio, screen["cost_ratio_head_over_R"], "stage-one reported cost ratio", atol=1e-12)
    g = screen["g_head"]
    close(screen["R"]["doc_mean_nll"] - screen["head"]["doc_mean_nll"], g["mean"], "stage-one mean gain")
    gate = {"technical": stage1["technical_checks"]["passed"], "cost_ratio": 0.98 <= ratio <= 1.02,
            "effect": g["mean"] >= 0.005 and g["lower_one_sided_90"] > 0}
    require(gate == stage1["gate"] and not all(gate.values()) and stage1["decision"] == "fail",
            "Original stage-one no-go does not reproduce")
    path = OUT32 / "stage1_per_document.pt"
    per_document = {"available": path.exists()}
    if path.exists():
        data = torch.load(path, map_location="cpu", weights_only=False)
        require(len(data["R"]) == len(data["head"]) == stage1["documents"]["stage1_screen"] == 367,
                "Stage-one per-document count mismatch")
        for name in ("R", "head"):
            close(sum(data[name]) / len(data[name]), screen[name]["doc_mean_nll"], f"stage-one {name} NLL")
        weight = stage1["calibration"]["head"]["bracket"]["weight_high"]
        require(len(data["head_high"]) == len(data["head_low"]) == len(data["head"]), "Head component lengths differ")
        for a, b, got in zip(data["head_high"], data["head_low"], data["head"]):
            close(weight * a + (1 - weight) * b, got, "stage-one per-document mixture", atol=1e-12)
        fresh = R.bootstrap([r - h for r, h in zip(data["R"], data["head"])])
        for key in ("mean", "sd", "lower_one_sided_90", "lower_one_sided_95"):
            close(fresh[key], g[key], f"stage-one bootstrap {key}", atol=1e-12)
        for fresh_value, saved_value in zip(fresh["ci95"], g["ci95"]):
            close(fresh_value, saved_value, "stage-one bootstrap CI", atol=1e-12)
        per_document.update({"sha256": C.sha256_file(path), "documents": len(data["R"]),
                             "nll_and_mixture_reproduced": True, "bootstrap": fresh})
    return {"brackets": bracket_checks, "cost_ratio": ratio_check,
            "cost_evidence_scope": "Ratio independently recomputed from reported aggregate costs; "
                                   "original per-document artifact stores NLL only, so FLOPs are not independently re-executed.",
            "per_document": per_document, "gate": gate, "original_decision": "fail", "no_go_preserved": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-documents", type=int, default=16)
    parser.add_argument("--threads", type=int, default=min(4, os.cpu_count() or 1))
    args = parser.parse_args()
    require(1 <= args.batch_documents <= 32, "Use 1--32 documents per batch to bound memory")
    require(1 <= args.threads <= 8, "Use 1--8 CPU threads")
    torch.set_num_threads(args.threads)
    started = time.perf_counter()
    cap = float(C.CONFIG.get("head_audit_cap_seconds", 1800))
    guard = lambda: C.check_time(started, cap)
    C.verify_inputs()
    paths = {"head": OUT32 / "head.pt", "head_receipt": RESULTS32 / "head.json",
             "stage1_receipt": RESULTS32 / "stage1.json", "tables": OUT32 / "tables_readout_train.pt",
             "tables_receipt": RESULTS32 / "tables.json"}
    hashes = {name: C.sha256_file(path) for name, path in paths.items()}
    blob = torch.load(paths["head"], map_location="cpu", weights_only=False)
    head_receipt = json.loads(paths["head_receipt"].read_text(encoding="utf-8"))
    stage1 = json.loads(paths["stage1_receipt"].read_text(encoding="utf-8"))
    tables_receipt = json.loads(paths["tables_receipt"].read_text(encoding="utf-8"))
    theta_sha = C.CONFIG["input_sha256"]["artifacts/chunk-depth-v31/pilot_SCHED-A_s17.pt"]
    require(tables_receipt["theta_s_sha256"] == stage1["theta_s_sha256"] == theta_sha,
            "Existing tables or stage-one result identify a different theta_S")
    require(tables_receipt["roles"]["readout_train"]["documents"] == 4111
            and tables_receipt["roles"]["readout_train"]["lines_scored"] == 14703,
            "Theta-S table receipt has unexpected dimensions")
    selection = audit_selection(blob, head_receipt, stage1)
    original = audit_stage1(stage1)
    guard()
    items = load_role("readout_train")
    tables = torch.load(paths["tables"], map_location="cpu", weights_only=False)
    require(len(items) == len(tables) == 4111, "Theta-S table count mismatch")
    for item, table in zip(items, tables):
        n_lines = len(doc_structure(item))
        require(tuple(table["reps"].shape) == (5, n_lines, 5280)
                and table["reps"].dtype == torch.bfloat16, "Invalid theta-S boundary table shape/dtype")
        require(tuple(table["line_nll"].shape) == (n_lines, 5), "Invalid theta-S NLL table shape")
    table_bytes = sum(value.numel() * value.element_size() for table in tables for value in table.values())
    require(table_bytes < 1024**3, "Unexpected table size; refuse an unbounded CPU allocation")
    normalization = audit_normalization(items, tables, blob, head_receipt, args.batch_documents, guard)
    objective = audit_objective(items, tables, blob, head_receipt, normalization["weight_sum"], args.batch_documents, guard)
    cost = audit_greedy_cost(items, tables, blob, selection["selected_fit"]["greedy_relative_cost"], guard)
    frozen = audit_frozen_head(blob, tables[0], items[0])
    guard()
    require(C.sha256_file(paths["head"]) == hashes["head"], "Saved head changed during audit")
    payload = C.record("head_reuse_audit", status="passed", source_artifact_sha256=hashes,
                       table_source_theta_s_sha256=theta_sha,
                       selection=selection, normalization=normalization, objective=objective,
                       sequential_greedy_cost=cost, frozen_head=frozen, original_stage1=original,
                       execution={"device": "cpu", "threads": args.threads,
                                  "batch_documents": args.batch_documents, "table_tensor_bytes": table_bytes,
                                  "full_design_matrix_allocated": False, "new_lbfgs_training": False,
                                  "stage2_screen_opened": False},
                       interpretation="Completed frozen-theta_S readout is reusable for the authorized mechanism follow-up; "
                                      "the original stage-one no-go remains unchanged.",
                       seconds=time.perf_counter() - started)
    C.write_json(C.RESULTS / "head_reuse.json", payload)
    print(json.dumps({"status": "passed", "selection": {key: selection[key] for key in ("lambda", "gamma", "tau")},
                      "objective": objective, "normalization": normalization,
                      "frozen_head": frozen, "stage1_no_go_preserved": True}, indent=2), flush=True)


if __name__ == "__main__":
    main()
