"""CPU 独立复核完成的机制诊断回执、配对训练和逐篇评价。

所有必需结果须已存在。缺项、不匹配或预算超限立即失败；只在全部检查通过后
独占创建 validation.json。复算统计量，不加载模型，不执行 CUDA。
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

import torch

from experiments.chunk_depth_v3.analyze_survey import position_cost
from . import common as C

ARMS = ("E-G", "E-C")
UPDATES = (0, 23, 46, 92)
DEPTHS = (4, 6, 8, 12, 16)


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def close(a, b, label, atol=1e-10, rtol=0):
    a, b = float(a), float(b)
    require(math.isfinite(a) and math.isfinite(b) and math.isclose(a, b, abs_tol=atol, rel_tol=rtol),
            f"{label}: {a} != {b}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_pt(path):
    return torch.load(path, map_location="cpu", weights_only=False)


def vector_close(a, b, label, atol=1e-10):
    require(len(a) == len(b), f"{label}: lengths differ")
    for i, (x, y) in enumerate(zip(a, b)):
        close(x, y, f"{label}[{i}]", atol=atol)


def bootstrap(values):
    """独立重写预登记的文档 bootstrap，不复用实验的汇总函数。"""
    x = torch.tensor(values, dtype=torch.float64, device="cpu")
    require(x.ndim == 1 and len(x) > 1 and bool(torch.isfinite(x).all()), "Invalid bootstrap values")
    config = C.CONFIG["evaluation"]["bootstrap"]
    rng = torch.Generator(device="cpu").manual_seed(config["seed"])
    idx = torch.randint(len(x), (config["repeats"], len(x)), generator=rng, device="cpu")
    means = x[idx].mean(1)
    return {"mean": float(x.mean()), "sd": float(x.std()), "n": len(x),
            "ci95": [float(torch.quantile(means, q)) for q in (0.025, 0.975)],
            "lower_one_sided_90": float(torch.quantile(means, 0.1)),
            "lower_one_sided_95": float(torch.quantile(means, 0.05))}


def check_bootstrap(values, saved, label):
    actual = bootstrap(values)
    require(actual["n"] == saved["n"], f"{label}: n mismatch")
    for key in ("mean", "sd", "lower_one_sided_90", "lower_one_sided_95"):
        close(actual[key], saved[key], f"{label}.{key}", atol=1e-12)
    vector_close(actual["ci95"], saved["ci95"], f"{label}.ci95", atol=1e-12)
    return {**actual, "mean_sign": (actual["mean"] > 0) - (actual["mean"] < 0),
            "ci_entirely_positive": actual["ci95"][0] > 0,
            "ci_entirely_negative": actual["ci95"][1] < 0}


def validate_cross(receipt):
    cells = read_pt(C.OUT / "cross_cells.pt")
    require(set(cells) == {"S_S", "S_E", "E_S", "E_E"}, "Missing crossed cell")
    require(receipt["documents"] == 505, "Cross document count")
    for key, cell in cells.items():
        require(len(cell["doc"]) == 505, f"Cross {key} does not have 505 documents")
        mean = sum(cell["doc"]) / 505
        close(mean, cell["mean"], f"cross {key} artifact mean")
        close(mean, receipt["cells"][key]["mean"], f"cross {key} receipt mean")
        close(cell["cost"], receipt["cells"][key]["cost"], f"cross {key} cost", atol=1e-12)
        require(cell["cost"] > 0, f"cross {key}: nonpositive cost")
    old = read_json(C.ROOT / "results/chunk-depth-v31/decompose.json")
    for schedule, old_key in (("S", "SCHED-A"), ("E", "E-static")):
        close(cells[f"E_{schedule}"]["cost"], cells[f"S_{schedule}"]["cost"],
              f"fixed cross schedule {schedule}", atol=1e-12)
        close(cells[f"{schedule}_{schedule}"]["cost"], old["policies"][old_key]["relative_cost"],
              f"original diagonal cost {schedule}", atol=1e-12)
    differences = {}
    for schedule in ("S", "E"):
        differences[f"E_minus_S_fixed_schedule_{schedule}"] = [
            e - s for e, s in zip(cells[f"E_{schedule}"]["doc"], cells[f"S_{schedule}"]["doc"])]
    for theta in ("S", "E"):
        differences[f"schedule_E_minus_S_fixed_theta_{theta}"] = [
            e - s for e, s in zip(cells[f"{theta}_E"]["doc"], cells[f"{theta}_S"]["doc"])]
    differences["interaction"] = [ee-es-se+ss for ee, es, se, ss in
        zip(cells["E_E"]["doc"], cells["E_S"]["doc"], cells["S_E"]["doc"], cells["S_S"]["doc"])]
    return {"documents": 505, "effects": {k: check_bootstrap(v, receipt["effects"][k], k)
                                           for k, v in differences.items()},
            "cost_scope": "Cross artifacts retain aggregate cost only: checked equal columns and original diagonal receipts; no raw cross FLOPs reconstruction claimed."}


def validate_training(receipts, identities):
    final, refs = {}, {}
    source = read_pt(C.ROOT / "artifacts/chunk-depth-v31/pilot_SCHED-A_s17.pt")["lora"]
    snap = read_pt(C.OUT / "snapshot_pilot_E-G_s17_u000.pt")
    require(snap["arm"] == "E-G" and snap["update"] == 0 and not snap["opt"]["state"],
            "Initial snapshot has wrong arm, update or nonempty optimizer state")
    require(len(source) == len(snap["lora"]) and all(torch.equal(a, b) for a, b in zip(source, snap["lora"])),
            "E-G initial LoRA differs elementwise from theta_S")
    for arm in ARMS:
        receipt = receipts[f"pilot_{arm}_s17"]
        require(receipt["arm"] == arm and receipt["updates"] == 93 and receipt["documents"] == 373
                and receipt["smoke"] is False and receipt["head_frozen"] is True, f"{arm}: training metadata")
        path = C.OUT / f"pilot_{arm}_s17.pt"
        require(C.sha256_file(path) == receipt["checkpoint_sha256"], f"{arm}: checkpoint hash")
        final[arm] = read_pt(path)
        blob = final[arm]
        require(blob["arm"] == arm and blob["update"] == 93 and blob["identities"] == identities,
                f"{arm}: checkpoint metadata")
        states = blob["opt"]["state"]
        require(len(states) == len(source) and all(float(s["step"]) == 93 for s in states.values()),
                f"{arm}: AdamW step counts")
        require(all(bool(torch.isfinite(p).all()) for p in blob["lora"]), f"{arm}: nonfinite parameters")
        refpath = C.OUT / f"pilot_{arm}_s17_reference_logp.pt"
        require(Path(receipt["reference_logp"]).resolve() == refpath.resolve(), f"{arm}: reference path")
        refs[arm] = read_pt(refpath)
        require(refs[arm]["identities"] == identities and len(refs[arm]["logp"]) == 373,
                f"{arm}: reference identity/count")
        require(len(receipt["log"]) == 93, f"{arm}: training log count")
        for update, entry in enumerate(receipt["log"]):
            require(entry["update"] == update and entry["batch_indices"] == blob["data_order"][update*8:(update+1)*8].tolist(),
                    f"{arm}: training data order at {update}")
            require(entry["actual_parameter_delta_norm"] > 0, f"{arm}: empty parameter update {update}")
            expected = entry["gen"] + (entry["ctrl"] if arm == "E-C" else 0)
            close(entry["optimized_loss"], expected, f"{arm}: optimized loss {update}")
        for name, digest in receipt["training_source_sha256"].items():
            require(C.sha256_file(C.ROOT / "artifacts/chunk-depth-v31" / name) == digest,
                    f"{arm}: training source {name}")
        require(C.sha256_file(C.ROOT / "results/chunk-depth-v31/kv.json") == receipt["kv_source_sha256"],
                f"{arm}: KV source changed")
    require(torch.equal(final["E-G"]["data_order"], final["E-C"]["data_order"]), "Arm data orders differ")
    rng = torch.Generator(device="cpu").manual_seed(17)
    expected_order = torch.cat([torch.randperm(373, generator=rng) for _ in range(2)])
    require(torch.equal(final["E-G"]["data_order"], expected_order), "Data order differs from seed 17 recipe")
    require(all(torch.equal(g, c) for g, c in zip(refs["E-G"]["logp"], refs["E-C"]["logp"])),
            "Arm fixed references differ elementwise")
    first = {arm: receipts[f"pilot_{arm}_s17"]["log"][0] for arm in ARMS}
    require(first["E-G"]["batch_indices"] == first["E-C"]["batch_indices"] == snap["batch_indices"],
            "First batches differ")
    close(first["E-G"]["gen"], first["E-C"]["gen"], "Initial generation loss", atol=1e-10)
    for key in ("generation_weights_document_normalized", "generation_weights_token_pooled",
                "sampled_weights_token_pooled", "sampled_weights_line_pooled"):
        vector_close(first["E-G"][key], first["E-C"][key], f"Initial {key}", atol=1e-12)
    return {"initial_parameters_equal_theta_S": True, "fixed_references_tensor_equal": True,
            "data_order_equal_and_reproduced": True, "updates": 93, "optimizer_steps": 93,
            "first_generation_loss": first["E-G"]["gen"], "first_batch": first["E-G"]["batch_indices"]}


def check_bracket(record, label):
    br = record["bracket"]
    require(isinstance(br, dict), f"{label}: missing budget bracket")
    points = {row["param"]: row for row in record["online_points"]}
    hi, lo = points[br["param_high_cost"]], points[br["param_low_cost"]]
    w = br["weight_high"]
    require(0 <= w <= 1 and lo["relative_cost"] <= 0.5 <= hi["relative_cost"], f"{label}: invalid bracket")
    cost = w*hi["relative_cost"] + (1-w)*lo["relative_cost"]
    close(cost, 0.5, f"{label}: calibration cost", atol=1e-12)
    close(w*hi["doc_mean_nll"] + (1-w)*lo["doc_mean_nll"], record["cal_matched_nll"],
          f"{label}: calibrated NLL")
    return cost


def validate_run(run, identities):
    require(len(run["stats"]) == len(run["rows"]) == len(identities), "Endpoint document counts differ")
    for entry, row, item in zip(run["stats"], run["rows"], identities):
        require((row["split"], row["row"]) == (item["split"], item["row"]), "Endpoint row order changed")
        chunks, length = item["chunk_of_pos"], len(item["ids"])
        lines = [[t for t in range(length) if chunks[t] == c] for c in range(1, item["n_chunks"])]
        require(len(entry["depths"]) == len(lines), "Endpoint depth count mismatch")
        require(all(d in DEPTHS for d in entry["depths"]), "Invalid depth")
        actual_cost = sum(position_cost(t, depth) for line, depth in zip(lines, entry["depths"]) for t in line)
        base16 = sum(position_cost(t, 16) for line in lines for t in line)
        close(actual_cost, entry["cost"], "Recomputed endpoint FLOPs", atol=1e-3)
        close(base16, entry["base16"], "Recomputed endpoint base16", atol=1e-3)
        n = sum(t < length-1 for line in lines for t in line)
        require(n == entry["tokens"] and n > 0, "Endpoint scored-token count")
        close(entry["sum_nll"] / n, entry["mean_nll"], "Endpoint NLL average")


def validate_evaluation(cal, receipt):
    require(cal["status"] == "locked" and cal["documents"] == 612 and cal["screen_opened"] is False,
            "Calibration was not locked before screen")
    cal_sha = C.sha256_file(C.RESULTS / "calibration.json")
    require(receipt["calibration_sha256"] == cal_sha, "Evaluation calibration hash")
    head_sha = C.CONFIG["input_sha256"]["artifacts/chunk-depth-v32/head.pt"]
    require(cal["head_sha256"] == receipt["head_sha256"] == head_sha, "Frozen head hash")
    for arm in ARMS:
        digest = C.sha256_file(C.OUT / f"pilot_{arm}_s17.pt")
        require(cal["checkpoint_sha256"][arm] == receipt["checkpoint_sha256"][arm] == digest,
                f"{arm}: locked checkpoint hash")
        check_bracket(cal["calibration"][arm], arm)
    stage1 = read_json(C.ROOT / "results/chunk-depth-v32/stage1.json")
    fixed = cal["fixed_schedule"]
    require(fixed == receipt["fixed_schedule"], "Fixed schedule metadata differs")
    require(fixed["ridge_sha256"] == C.sha256_file(C.ROOT / "artifacts/chunk-depth-v32/R.pt"), "R hash")
    require(fixed["source_stage1_sha256"] == C.sha256_file(C.ROOT / "results/chunk-depth-v32/stage1.json"),
            "Stage-one schedule source hash")
    require(fixed["bracket"] == stage1["calibration"]["R"]["bracket"], "Fixed R bracket changed")
    check_bracket(stage1["calibration"]["R"], "fixed R")
    data = read_pt(C.OUT / "evaluation_per_document.pt")
    require(data["calibration_sha256"] == cal_sha, "Per-document calibration hash")
    ids = data["identities"]
    require(receipt["documents"] == len(ids) == 378, "Screen must have 378 documents")
    require(len({(i["split"], i["row"]) for i in ids}) == 378, "Duplicate screen document")
    cal_runs = read_pt(C.OUT / "calibration_runs.pt")
    require(len(cal_runs) >= 4, "Calibration endpoint runs missing")
    cal_ids = {(r["split"], r["row"]) for r in cal_runs[0]["rows"]}
    require(len(cal_ids) == 612 and not cal_ids.intersection((i["split"], i["row"]) for i in ids),
            "Calibration/screen overlap or wrong calibration count")
    for arm in ARMS:
        arm_runs = [r for r in cal_runs if r["label"] == f"calibration {arm}"]
        unmatched = list(cal["calibration"][arm]["online_points"])
        require(len(arm_runs) == len(unmatched), f"{arm}: calibration run/point count")
        for run in arm_runs:
            rows = run["stats"]
            require(len(rows) == 612 and run["rows"] == cal_runs[0]["rows"],
                    f"{arm}: calibration document order/count")
            cost = sum(r["cost"] for r in rows) / sum(r["base16"] for r in rows)
            mean = sum(r["mean_nll"] for r in rows) / len(rows)
            matches = [p for p in unmatched if abs(p["relative_cost"]-cost) <= 1e-12
                       and abs(p["doc_mean_nll"]-mean) <= 1e-10]
            require(bool(matches), f"{arm}: recorded calibration point does not reproduce raw rows")
            unmatched.remove(matches[0])
    fixed_runs = {}
    for arm in ARMS:
        for category, label_suffix, receipt_key in (("head", "head", "policies"),
                ("fixed_schedule", "fixed R", "fixed_schedule_policies")):
            ev = data[category][arm]
            saved = receipt[receipt_key][arm]
            require(len(ev["doc"]) == 378, "Policy per-document count")
            close(sum(ev["doc"])/378, saved["doc_mean_nll"], f"{arm} {category}: mean")
            close(ev["cost"], saved["relative_cost"], f"{arm} {category}: cost", atol=1e-12)
            runs = [r for r in data["runs"] if r["label"] == f"stage2 {arm} {label_suffix}"]
            require(len(runs) == 2, f"{arm} {category}: expected two endpoint runs")
            for run in runs:
                validate_run(run, ids)
            if category == "fixed_schedule":
                fixed_runs[arm] = runs
                w = fixed["bracket"]["weight_high"]
            else:
                w = cal["calibration"][arm]["bracket"]["weight_high"]
            high, low = [r["stats"] for r in runs]
            mixed = [w*a["mean_nll"]+(1-w)*b["mean_nll"] for a, b in zip(high, low)]
            mixed_cost = sum(w*a["cost"]+(1-w)*b["cost"] for a, b in zip(high, low))/sum(a["base16"] for a in high)
            vector_close(mixed, ev["doc"], f"{arm} {category}: mixed NLL", atol=1e-12)
            close(mixed_cost, ev["cost"], f"{arm} {category}: mixed cost", atol=1e-12)
    for eg, ec in zip(fixed_runs["E-G"], fixed_runs["E-C"]):
        for g, c in zip(eg["stats"], ec["stats"]):
            require(g["depths"] == c["depths"], "Fixed R endpoint depths differ across arms")
            close(g["cost"], c["cost"], "Fixed R endpoint cost", atol=1e-3)
    close(data["fixed_schedule"]["E-C"]["cost"]-data["fixed_schedule"]["E-G"]["cost"], 0,
          "Fixed R mixed cost difference", atol=1e-12)
    close(receipt["fixed_schedule_cost_difference"], 0, "Reported fixed R cost difference", atol=1e-12)
    gain = [g-c for g, c in zip(data["head"]["E-G"]["doc"], data["head"]["E-C"]["doc"])]
    harm = [c-g for c, g in zip(data["fixed_schedule"]["E-C"]["doc"], data["fixed_schedule"]["E-G"]["doc"])]
    bg = check_bootstrap(gain, receipt["g_joint_EG_minus_EC"], "g_joint EG-EC")
    bh = check_bootstrap(harm, receipt["h_gen_EC_minus_EG"], "h_gen EC-EG")
    ratio = data["head"]["E-C"]["cost"] / data["head"]["E-G"]["cost"]
    close(ratio, receipt["cost_ratio_EC_over_EG"], "Online cost ratio", atol=1e-12)
    flags = {"same_cost": 0.98 <= ratio <= 1.02, "point_at_least_0_005": bg["mean"] >= .005,
             "one_sided_90_lower_positive": bg["lower_one_sided_90"] > 0}
    require(flags == receipt["diagnostic_thresholds"] and all(flags.values()) == receipt["diagnostic_thresholds_all_met"],
            "Diagnostic threshold calculation differs")
    return {"documents": 378, "calibration_documents": 612, "calibration_hash": cal_sha,
            "g_joint": bg, "h_gen": bh, "cost_ratio": ratio, "diagnostic_thresholds": flags,
            "fixed_R_endpoint_depths_and_costs_equal": True, "endpoint_FLOPs_recomputed": True}


def validate_gradients(receipt, identities):
    require(receipt["completed"] is True and [s["update"] for s in receipt["snapshots"]] == list(UPDATES),
            "Gradient diagnostics incomplete")
    require(receipt["identities"] == identities, "Gradient diagnostic input identities")
    require(C.sha256_file(C.OUT / "gradient_diag_fixed_inputs.pt") == receipt["fixed_inputs_sha256"],
            "Gradient fixed-input hash")
    fixed = read_pt(C.OUT / "gradient_diag_fixed_inputs.pt")
    require(fixed["identities"] == identities and
            [s["update"] for s in fixed["snapshots"]] == list(UPDATES), "Gradient fixed-input identities")
    for key, name in (("items", "items_warm.pt"), ("features", "features_warm.pt")):
        digest = C.sha256_file(C.ROOT / "artifacts/chunk-depth-v31" / name)
        require(receipt["data_source_sha256"][key] == fixed["data_source_sha256"][key] == digest,
                f"Gradient diagnostic {key} source hash")
    max_replay = 0.0
    for entry in receipt["snapshots"]:
        path = C.OUT / f"snapshot_pilot_E-G_s17_u{entry['update']:03d}.pt"
        require(C.sha256_file(path) == entry["sha256"], "Gradient snapshot hash")
        require(len(entry["batches"]) == 4, "Gradient diagnostic batches missing")
        for block in [*entry["batches"], entry["eight_document_mean"]]:
            replay = float(block["adamw"]["fixed_replay_minus_training_loss"])
            require(math.isfinite(replay) and abs(replay) <= 1e-7, "Frozen generation replay drift exceeds 1e-7")
            max_replay = max(max_replay, abs(replay))
            require(set(block["adamw"]["branches"]) == {"G", "G_plus_C"}, "Cloned optimizer branch missing")
    return {"updates": list(UPDATES), "max_absolute_fixed_replay_error": max_replay}


def main():
    names = ("cross", "head_reuse", "math_check", "pilot_E-G_s17", "pilot_E-C_s17",
             "calibration", "evaluation", "gradient_diag")
    paths = [C.RESULTS / f"{n}.json" for n in names]
    paths += [C.OUT / name for name in ("cross_cells.pt", "pilot_E-G_s17.pt", "pilot_E-C_s17.pt",
        "pilot_E-G_s17_reference_logp.pt", "pilot_E-C_s17_reference_logp.pt", "calibration_runs.pt",
        "evaluation_per_document.pt", "gradient_diag_fixed_inputs.pt")]
    paths += [C.OUT / f"snapshot_pilot_E-G_s17_u{u:03d}.pt" for u in UPDATES]
    missing = [str(p) for p in paths if not p.is_file()]
    if missing:
        raise FileNotFoundError("Required completed results are missing:\n" + "\n".join(missing))
    target = C.RESULTS / "validation.json"
    require(not target.exists(), "validation.json already exists; refusing overwrite")
    identities = C.verify_inputs()
    receipts = {n: read_json(C.RESULTS / f"{n}.json") for n in names}
    config_sha = C.sha256_file(C.CONFIG_PATH)
    for name, row in receipts.items():
        if name != "math_check":
            require(row["config_sha256"] == config_sha and row["input_sha256"] == identities,
                    f"{name}: frozen config/input hash mismatch")
    head, math_receipt = receipts["head_reuse"], receipts["math_check"]
    require(head["status"] == "passed" and head["original_stage1"]["no_go_preserved"] is True,
            "Head audit failed or original no-go changed")
    require(head["source_artifact_sha256"]["head"] == identities["artifacts/chunk-depth-v32/head.pt"], "Head audit digest")
    require(math_receipt["status"] == "passed" and math_receipt["device"] == "cpu" and math_receipt["cases"] == 16,
            "Math receipt incomplete")
    require(math_receipt["max_autograd_error"] <= 1e-10 and math_receipt["max_directional_error"] <= 1e-7
            and math_receipt["max_route_constant_shift_error"] <= 1e-10, "Math tolerances failed")
    require(len(math_receipt["records"]) == 16, "Math per-case records missing")
    for row in math_receipt["records"]:
        require(all(math.isfinite(x) and x <= 1e-10 for x in row["autograd_max_abs_error"].values()),
                "Math per-component autograd tolerance")
        require(all(math.isfinite(x) and x <= 1e-7 for x in row["central_difference_abs_error"].values()),
                "Math per-component directional tolerance")
    checks = {"cross": validate_cross(receipts["cross"]), "training": validate_training(receipts, identities),
              "evaluation": validate_evaluation(receipts["calibration"], receipts["evaluation"]),
              "gradient_diagnostics": validate_gradients(receipts["gradient_diag"], identities),
              "head_reuse_passed": True, "math_check_passed": True, "original_stage1_no_go_preserved": True}
    gpu_names = ["cross", "pilot_E-G_s17", "pilot_E-C_s17", "evaluation", "gradient_diag"]
    smoke_path = C.RESULTS / "smoke_E-C_s17.json"
    if smoke_path.exists():
        receipts["smoke_E-C_s17"] = read_json(smoke_path)
        gpu_names.append("smoke_E-C_s17")
        paths.append(smoke_path)
    elif (C.OUT / "smoke_E-C_s17.pt").exists():
        raise FileNotFoundError("Smoke checkpoint exists but its runtime receipt is missing")
    times = {name: float(receipts[name]["seconds"]) for name in gpu_names}
    require(all(math.isfinite(t) and t > 0 for t in times.values()), "Invalid task runtime")
    total = sum(times.values())
    cap = 3600 * float(C.CONFIG["gpu_hours_cap"])
    require(total < cap, f"GPU-task wall-time sum {total} >= cap {cap}")
    result = {"protocol_id": C.CONFIG["protocol"], "kind": "independent_completed_result_validation",
              "created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed", "device": "cpu",
              "config_sha256": config_sha, "input_sha256": identities,
              "source_sha256": {"validate_results.py": C.sha256_file(Path(__file__))},
              "checked_artifact_sha256": {str(p.relative_to(C.ROOT)): C.sha256_file(p) for p in paths},
              "checks": checks, "gpu_task_runtime": {"seconds_by_task": times, "total_seconds": total,
                  "total_hours": total/3600, "cap_seconds": cap, "within_cap": True,
                  "scope": "Conservative sum of complete wall times for recorded GPU tasks, including loading and pauses; CPU head/math/validation excluded, calibration already included in evaluation."}}
    with target.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(json.dumps({"status": "passed", "output": str(target), "gpu_task_hours": total/3600}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
