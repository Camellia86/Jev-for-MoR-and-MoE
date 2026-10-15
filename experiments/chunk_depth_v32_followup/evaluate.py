"""Calibrate the matched E-G/E-C arms, then open the reserved stage-two screen.

This is a mechanism follow-up, not a reversal of the v3.2 stage-one no-go.
All policy parameters are written to calibration.json before stage2_screen is loaded.
"""
from __future__ import annotations

from contextlib import contextmanager
import json
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.chunk_depth_v31 import run as R, screen as S
from experiments.chunk_depth_v31.data import OUT as OUT31
from experiments.chunk_depth_v32.common import OUT as OUT32, RESULTS as RESULTS32, load_role
from experiments.chunk_depth_v32.head import ridge_from_state
from experiments.chunk_depth_v32.stage1 import Head, head_policy, scores_head
from . import common as C


ARMS = ("E-G", "E-C")
B_FINE = [round(-20 + i * 0.05, 2) for i in range(801)]
CACHE_SCHEMA = "frozen-coda-inputs-v1"


def identity(item):
    """Bind cached rows to both the source row and the exact token/chunk sequence."""
    return {"split": item["split"], "row": int(item["row"]),
            "ids": list(item["ids"]), "chunk_of_pos": list(item["chunk_of_pos"]),
            "n_chunks": int(item["n_chunks"])}


def validate_features(items, features):
    if len(items) != len(features):
        raise RuntimeError("Calibration feature count does not match the item count")
    for item, feature in zip(items, features):
        tensor = feature.get("feats") if isinstance(feature, dict) else None
        expected = (len(R.V3), len(item["ids"]), 5280)
        if not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != expected:
            raise RuntimeError(f"Invalid coda-input shape for {item['split']}:{item['row']}: {expected}")
        if tensor.dtype != torch.bfloat16 or tensor.device.type != "cpu":
            raise RuntimeError("Cached coda inputs must be CPU BF16 tensors")


@torch.no_grad()
def calibration_features(model, items, start, cap):
    path = C.OUT / "features_calibration.pt"
    ids = [identity(item) for item in items]
    if path.exists():
        C.check_time(start, cap)
        blob = torch.load(path, map_location="cpu")
        if (not isinstance(blob, dict) or blob.get("schema") != CACHE_SCHEMA
                or blob.get("identities") != ids or tuple(blob.get("depths", ())) != R.V3
                or blob.get("input_sha256") != C.CONFIG["input_sha256"]):
            raise RuntimeError("Calibration cache identity, depths or fixed inputs have changed")
        validate_features(items, blob["features"])
        return blob["features"], {"path": str(path), "reused": True,
                                  "documents": len(items), "schema": CACHE_SCHEMA}

    # The v3.1 pilot-cal features are aligned with its persisted item list. Check
    # that list against this role before accepting their otherwise unlabelled rows.
    legacy_items = torch.load(OUT31 / "items_pilot_cal.pt", map_location="cpu")
    if [identity(item) for item in legacy_items] != ids[:len(legacy_items)]:
        raise RuntimeError("Legacy pilot-cal items do not match the calibration prefix")
    legacy = torch.load(OUT31 / "features_pilot_cal.pt", map_location="cpu")
    validate_features(legacy_items, legacy)
    features = [{"feats": feature["feats"]} for feature in legacy]
    del legacy
    for j in range(len(features), len(items)):
        C.check_time(start, cap)
        item = items[j]

        def extract():
            C.check_time(start, cap)
            token_ids = torch.tensor(item["ids"], device="cuda")[None]
            values = H.coda_inputs(model, token_ids, R.item_state(item, len(item["ids"])), R.V3)
            return {"feats": torch.stack([values[d][0] for d in R.V3]).cpu()}

        features.append(R.retry(extract))
        if (j + 1) % 100 == 0 or j + 1 == len(items):
            print(f"calibration features {j + 1}/{len(items)}", flush=True)
    validate_features(items, features)
    C.check_time(start, cap)
    blob = {"schema": CACHE_SCHEMA, "identities": ids, "depths": R.V3,
            "input_sha256": C.CONFIG["input_sha256"], "features": features,
            "legacy_documents": len(legacy_items)}
    temporary = path.with_suffix(".tmp.pt")
    torch.save(blob, temporary)
    temporary.replace(path)
    C.check_time(start, cap)
    return features, {"path": str(path), "reused": False, "documents": len(items),
                      "legacy_documents": len(legacy_items), "schema": CACHE_SCHEMA}


@contextmanager
def bounded_execution(start, cap, context, runs):
    """Instrument imported helpers only in this process, restoring them afterward."""
    old_run_all, old_predicted_cost = S.run_all, S.predicted_cost

    def run_all(model, items, policy, record=None, per_document=False):
        rows = []
        call = len(runs)
        label = context["label"]
        for j, item in enumerate(items):
            C.check_time(start, cap)
            pol = policy(j) if per_document else policy
            mark = None if record is None else len(record)

            def once():
                C.check_time(start, cap)
                if record is not None:
                    del record[mark:]
                return R.online(model, item, pol)

            nll, depths = R.retry(once)
            rows.append(R.doc_stats(item, nll, depths))
            if (j + 1) % 100 == 0 or j + 1 == len(items):
                print(f"{label} run={call} {j + 1}/{len(items)}", flush=True)
        C.check_time(start, cap)
        runs.append({"label": label, "call": call,
                     "rows": [{"split": item["split"], "row": int(item["row"])} for item in items],
                     "stats": rows})
        return rows

    def predicted_cost(scores, p):
        C.check_time(start, cap)
        return old_predicted_cost(scores, p)

    S.run_all, S.predicted_cost = run_all, predicted_cost
    try:
        yield
    finally:
        S.run_all, S.predicted_cost = old_run_all, old_predicted_cost


def checked_load(lora, path, digest, arm):
    if C.sha256_file(path) != digest:
        raise RuntimeError(f"Checkpoint changed after evaluation was started: {path}")
    blob = torch.load(path, map_location="cpu")
    values = blob.get("lora", [])
    if len(values) != len(lora.params) or any(v.shape != p.shape for v, p in zip(values, lora.params)):
        raise RuntimeError(f"Incompatible LoRA checkpoint: {path}")
    if blob.get("arm", arm) != arm:
        raise RuntimeError(f"Checkpoint arm does not match {arm}")
    lora.load(path)


def bracket_tuple(record):
    return (record["param_high_cost"], record["param_low_cost"], record["weight_high"])


def main():
    start = time.perf_counter()
    C.verify_inputs()
    cap = float(C.CONFIG.get("evaluation_cap_seconds", 9000))
    torch.backends.cuda.matmul.allow_tf32 = False
    checkpoints = {arm: C.OUT / f"pilot_{arm}_s17.pt" for arm in ARMS}
    digests = {arm: C.sha256_file(path) for arm, path in checkpoints.items()}
    head_path, ridge_path = OUT32 / "head.pt", OUT32 / "R.pt"
    head_sha, ridge_sha = C.sha256_file(head_path), C.sha256_file(ridge_path)
    stage1_path = RESULTS32 / "stage1.json"
    stage1_sha = C.sha256_file(stage1_path)
    head_blob = torch.load(head_path, map_location="cpu")
    head = Head(head_blob)
    ridge = ridge_from_state(torch.load(ridge_path, map_location="cpu"))
    stage1 = json.loads(stage1_path.read_text(encoding="utf-8"))
    fixed_bracket = stage1["calibration"]["R"]["bracket"]
    fixed = {"ridge_sha256": ridge_sha, "source_stage1_sha256": stage1_sha,
             "bracket": fixed_bracket}
    cal = load_role("calibration")
    if len(cal) != 612:
        raise RuntimeError(f"Unexpected calibration size: {len(cal)}")
    model = H.load_model()
    lora = S.Lora(model)
    # Frozen-core features do not depend on either arm's coda LoRA.
    features, feature_receipt = calibration_features(model, cal, start, cap)
    runs, timings, context = [], [], {"label": "calibration"}
    calibration = {}
    with bounded_execution(start, cap, context, runs):
        for arm in ARMS:
            checked_load(lora, checkpoints[arm], digests[arm], arm)
            tabs = []
            for j in range(0, len(cal), 25):
                C.check_time(start, cap)
                tabs.extend(R.retry(S.tables, model, cal[j:j + 25], features[j:j + 25]))
                if (j + 25) % 100 == 0 or j + 25 >= len(cal):
                    print(f"calibration tables {arm} {min(j + 25, len(cal))}/{len(cal)}", flush=True)
            scores = scores_head(cal, tabs, head)
            context["label"] = f"calibration {arm}"
            scan, br, curve = S.calibrate(model, cal, lambda b: head_policy(head, b),
                                          scores, B_FINE, arm, timings)
            calibration[arm] = S.scan_record(scan, br, curve)
            del tabs, scores, scan
            if br is None:
                payload = C.record("evaluation_calibration", status="no_budget_bracket",
                                   arm=arm, calibration=calibration, checkpoint_sha256=digests,
                                   head_sha256=head_sha, fixed_schedule=fixed,
                                   screen_opened=False, seconds=time.perf_counter() - start)
                C.write_json(C.RESULTS / "calibration.json", payload)
                torch.save(runs, C.OUT / "calibration_runs.pt")
                raise RuntimeError(f"No online budget bracket for {arm}; reserved screen remains unopened")

        del features
        lock = C.record("evaluation_calibration", status="locked", documents=len(cal),
                        checkpoint_sha256=digests, head_sha256=head_sha,
                        lambda_=head_blob["lambda"], fixed_schedule=fixed,
                        calibration=calibration, features=feature_receipt,
                        screen_opened=False, seconds=time.perf_counter() - start)
        C.write_json(C.RESULTS / "calibration.json", lock)
        torch.save(runs, C.OUT / "calibration_runs.pt")
        calibration_sha = C.sha256_file(C.RESULTS / "calibration.json")
        # No fitting or parameter choice may follow this boundary. Only now open
        # the previously unused stage-two screen and apply the recorded recipes.
        C.check_time(start, cap)
        screen = load_role("stage2_screen")
        if len(screen) != 378:
            raise RuntimeError(f"Unexpected stage-two screen size: {len(screen)}")
        if ({(it["split"], it["row"]) for it in screen}
                & {(it["split"], it["row"]) for it in cal}):
            raise RuntimeError("Calibration and reserved screen overlap")
        evals, fixed_evals = {}, {}
        runs.clear()
        for arm in ARMS:
            checked_load(lora, checkpoints[arm], digests[arm], arm)
            context["label"] = f"stage2 {arm} head"
            evals[arm] = S.evaluate_aligned(model, screen, lambda b: head_policy(head, b),
                                          bracket_tuple(calibration[arm]["bracket"]))
            context["label"] = f"stage2 {arm} fixed R"
            fixed_evals[arm] = S.evaluate_aligned(model, screen, lambda nu: S.position_policy(ridge, nu),
                                                bracket_tuple(fixed_bracket))
            torch.save({"identities": [identity(it) for it in screen], "head": evals,
                        "fixed_schedule": fixed_evals, "runs": runs,
                        "calibration_sha256": calibration_sha}, C.OUT / "evaluation_per_document.pt")

    fixed_cost_gap = fixed_evals["E-C"]["cost"] - fixed_evals["E-G"]["cost"]
    if abs(fixed_cost_gap) > 1e-12:
        raise RuntimeError("The same fixed schedule must have exactly the same cost for both LoRAs")
    if any(a["depths"] != b["depths"]
           for left, right in zip([r for r in runs if r["label"] == "stage2 E-G fixed R"],
                                  [r for r in runs if r["label"] == "stage2 E-C fixed R"])
           for a, b in zip(left["stats"], right["stats"])):
        raise RuntimeError("Fixed-schedule depth assignments differ across LoRAs")
    gain = R.bootstrap([g - c for g, c in zip(evals["E-G"]["doc"], evals["E-C"]["doc"])])
    generation_harm = R.bootstrap([c - g for c, g in zip(fixed_evals["E-C"]["doc"],
                                                       fixed_evals["E-G"]["doc"])])
    ratio = evals["E-C"]["cost"] / evals["E-G"]["cost"]
    diagnostic = {"same_cost": 0.98 <= ratio <= 1.02,
                  "point_at_least_0_005": gain["mean"] >= 0.005,
                  "one_sided_90_lower_positive": gain["lower_one_sided_90"] > 0}
    C.check_time(start, cap)
    payload = C.record("matched_training_evaluation", documents=len(screen),
                       checkpoint_sha256=digests, head_sha256=head_sha,
                       calibration_sha256=calibration_sha, fixed_schedule=fixed,
                       policies={arm: S.report(evals[arm]) for arm in ARMS},
                       fixed_schedule_policies={arm: S.report(fixed_evals[arm]) for arm in ARMS},
                       g_joint_EG_minus_EC=gain, h_gen_EC_minus_EG=generation_harm,
                       cost_ratio_EC_over_EG=ratio, fixed_schedule_cost_difference=fixed_cost_gap,
                       diagnostic_thresholds=diagnostic,
                       diagnostic_thresholds_all_met=all(diagnostic.values()),
                       interpretation="Positive g_joint favors E-C under separately calibrated policies; "
                                      "positive h_gen means E-C has higher NLL at an identical fixed schedule. "
                                      "These diagnostics do not change the v3.2 stage-one no-go.",
                       seconds_per_online_run=sum(timings) / len(timings),
                       seconds=time.perf_counter() - start)
    C.write_json(C.RESULTS / "evaluation.json", payload)
    print(json.dumps({k: payload[k] for k in ("g_joint_EG_minus_EC", "h_gen_EC_minus_EG",
                                            "cost_ratio_EC_over_EG", "diagnostic_thresholds")},
                     indent=2), flush=True)


if __name__ == "__main__":
    main()
