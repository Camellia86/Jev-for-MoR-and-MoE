"""Frozen-surrogate gradients and cloned-AdamW counterfactual steps.

At a saved parameter vector theta_0, sample each diagnostic history once and
save q_cd = stopgrad(p_cd(theta_0)).  Every before/after generation evaluation
then uses G_0(theta) = mean_j sum_cd q_jcd L_jcd(theta) / N_j.
Thus g_G = grad G_0(theta_0), and g_G dot Delta is the directional first-order
prediction for the actual AdamW displacement.  Neither the observed finite
step nor this local surrogate is an estimate of the online sequential return.

The four two-document batches and their eight-document mean are distinguished:
cos(mean g_G, mean g_C) is calculated from the averaged vectors, never from an
average of the four cosines.  Weighted CE/Brier/KL gradients match training.
"""

from __future__ import annotations

import copy
import json
import math
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import sha256_file
from experiments.shared_depth_v2.policy import lr_at
from experiments.chunk_depth_v31 import train as T31
from experiments.chunk_depth_v32_followup import common as C
from experiments.chunk_depth_v32_followup import train as T


COMPONENTS = ("gen", "route", "ce", "brier", "kl")
UPDATES = (0, 23, 46, 92)
DIAGNOSTIC_SEED = 20261013


def dot(a, b):
    return sum(float((x.double() * y.double()).sum()) for x, y in zip(a, b))


def norm(a):
    return math.sqrt(max(0.0, dot(a, a)))


def add(*vectors):
    return [sum(parts) for parts in zip(*vectors)]


def scaled(vector, factor):
    return [x * factor for x in vector]


def gradient_summary(grads):
    """All summaries use pooled vectors with coefficients already applied."""
    pooled = dict(grads)
    pooled["ctrl"] = add(*(grads[k] for k in COMPONENTS[1:]))
    pooled["joint"] = add(grads["gen"], pooled["ctrl"])
    ng = norm(grads["gen"])
    details = {}
    for name, vector in pooled.items():
        nv = norm(vector)
        inner = dot(grads["gen"], vector)
        details[name] = {
            "norm": nv, "dot_with_generation": inner,
            "cosine_with_generation": inner / (ng * nv) if ng and nv else None,
            "norm_ratio_to_generation": nv / ng if ng else None,
        }
    names = list(pooled)
    return {
        "components": details,
        "gram_order": names,
        "gram": [[dot(pooled[a], pooled[b]) for b in names] for a in names],
        "coefficients": {"route": 1.0, "ce": 0.1, "brier": 0.01, "kl": 0.1},
        "definition": "vectors averaged over documents before dot/norm/cosine",
    }, pooled


@torch.no_grad()
def restore_parameters(params, values):
    if len(params) != len(values):
        raise ValueError("LoRA parameter count changed")
    for p, value in zip(params, values):
        if p.shape != value.shape:
            raise ValueError("LoRA parameter shape changed")
        p.copy_(value.to(device=p.device, dtype=p.dtype))
        p.grad = None


def batch_gradients(model, params, items, feats, refs, head, indices, guard):
    """Differentiate the training objective, then retain only CPU data."""
    gradients = {name: [torch.zeros_like(p, device="cpu") for p in params]
                 for name in COMPONENTS}
    contexts, losses = [], {name: 0.0 for name in (*COMPONENTS, "ctrl")}
    for j in indices:
        guard()
        # The same seed is used for document j at all four saved updates.
        seed = T.history_seed(DIAGNOSTIC_SEED, 0, j)
        gen, ctrl, stats = T.loss_for_doc(model, items[j], feats[j], head, refs[j], seed)
        if not bool(torch.isfinite(gen + ctrl)):
            raise FloatingPointError(f"Nonfinite diagnostic loss, document {j}")
        losses["ctrl"] += float(ctrl.detach()) / len(indices)
        for position, name in enumerate(COMPONENTS):
            value = stats["component_losses"][name]
            losses[name] += float(value.detach()) / len(indices)
            values = torch.autograd.grad(value, params, retain_graph=position < len(COMPONENTS) - 1)
            for accumulator, value in zip(gradients[name], values):
                if not bool(torch.isfinite(value).all()):
                    raise FloatingPointError(f"Nonfinite {name} gradient, document {j}")
                accumulator.add_(value.detach().cpu(), alpha=1.0 / len(indices))
            del values
        contexts.append({
            "document_index": j, "history_seed": seed,
            "sampled_actions": stats["sampled_actions"],
            "logp": stats["policy_logp"].detach().cpu().clone(),
            "probability": stats["policy_logp"].detach().cpu().exp(),
            "n_tokens": stats["n_tokens"],
        })
        del gen, ctrl, stats
    return gradients, contexts, losses


@torch.no_grad()
def fixed_generation(model, items, feats, contexts, guard):
    """Re-evaluate G_0 using the baseline probabilities, without resampling."""
    values = []
    for context in contexts:
        guard()
        j = context["document_index"]
        item, f = items[j], feats[j]
        device = next(model.parameters()).device
        ids = torch.tensor(item["ids"], device=device)
        fc = H.freqs(model, torch.arange(len(ids)))
        st = T31.doc_structure(item)
        q = context["probability"].to(device=device, dtype=torch.float64)
        if q.shape != (len(st), T.K) or not bool(torch.isfinite(q).all()):
            raise ValueError("Fixed probability shape or values invalid")
        n = sum(line["n"] for line in st)
        if n != context["n_tokens"] or n <= 0:
            raise ValueError("Fixed generation token count changed")
        # Accumulate per-line losses in the same action order as training.
        action_nll = []
        for k in range(T.K):
            z = H.coda(model, f["feats"][k].to(device)[None], fc)[0]
            nll = torch.nn.functional.cross_entropy(
                H.head(model, z[None])[0, :-1], ids[1:], reduction="none")
            action_nll.append(nll)
        value = torch.zeros((), dtype=torch.float64, device=device)
        for i, line in enumerate(st):
            if line["n"]:
                lc = torch.stack([a[line["scored"]].sum() for a in action_nll])
                value += (q[i] * lc).sum() / n
        if not bool(torch.isfinite(value)):
            raise FloatingPointError("Nonfinite fixed generation loss")
        values.append(float(value))
    return sum(values) / len(values), values


def clone_step(model, params, snapshot, gradients, ggen, items, feats,
               contexts, baseline, guard):
    """One clipped AdamW step from exactly the saved optimizer state."""
    guard()
    restore_parameters(params, snapshot["lora"])
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    # load_state_dict mutates and places state on the parameter devices.
    opt.load_state_dict(copy.deepcopy(snapshot["opt"]))
    expected_lr = lr_at(int(snapshot["update"]), 93, 1e-4, 1e-5, 0.05)
    if any(abs(group["lr"] - expected_lr) > 1e-15 for group in opt.param_groups):
        raise ValueError("Saved AdamW learning rate differs from this update's schedule")
    for p, gradient in zip(params, gradients):
        p.grad = gradient.to(device=p.device, dtype=p.dtype).clone()
    raw_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
    opt.step()
    delta = [p.detach().cpu().double() - old.double()
             for p, old in zip(params, snapshot["lora"])]
    displacement = norm(delta)
    first_order = dot(ggen, delta)
    new_loss, per_document = fixed_generation(model, items, feats, contexts, guard)
    actual = new_loss - baseline
    result = {
        "lr": expected_lr, "gradient_norm_before_clip": raw_norm,
        "clip_coefficient": min(1.0, 1.0 / (raw_norm + 1e-6)),
        "actual_parameter_delta_norm": displacement,
        "generation_gradient_dot_delta": first_order,
        "generation_gradient_cosine_delta": first_order / (norm(ggen) * displacement)
            if norm(ggen) and displacement else None,
        "fixed_generation_before": baseline, "fixed_generation_after": new_loss,
        "fixed_generation_change": actual,
        "change_minus_first_order": actual - first_order,
        "per_document_after": per_document,
    }
    del opt, delta
    restore_parameters(params, snapshot["lora"])
    return result


def counterfactual_steps(model, params, snapshot, pooled, items, feats,
                         contexts, training_generation, guard):
    restore_parameters(params, snapshot["lora"])
    baseline, before = fixed_generation(model, items, feats, contexts, guard)
    if abs(baseline - training_generation) > 1e-7:
        raise RuntimeError(f"Fixed generation replay disagrees with training: {baseline} vs {training_generation}")
    branches = {}
    for name, gradient in (("G", pooled["gen"]), ("G_plus_C", pooled["joint"])):
        branches[name] = clone_step(model, params, snapshot, gradient, pooled["gen"],
                                   items, feats, contexts, baseline, guard)
    return {
        "per_document_before": before,
        "training_generation_loss": training_generation,
        "fixed_replay_minus_training_loss": baseline - training_generation,
        "branches": branches,
        "joint_minus_generation_only_loss_change":
            branches["G_plus_C"]["fixed_generation_change"] - branches["G"]["fixed_generation_change"],
        "joint_minus_generation_only_first_order":
            branches["G_plus_C"]["generation_gradient_dot_delta"] - branches["G"]["generation_gradient_dot_delta"],
    }


def main():
    started = time.perf_counter()
    cap = float(C.CONFIG.get("gradient_diagnostic_cap_seconds", 1800))
    guard = lambda: C.check_time(started, cap)
    identities = C.verify_inputs()
    report_path = C.RESULTS / "gradient_diag.json"
    data_path = C.OUT / "gradient_diag_fixed_inputs.pt"
    if report_path.exists() or data_path.exists():
        raise FileExistsError("Gradient diagnostic output already exists")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(T.SEED)
    torch.cuda.manual_seed_all(T.SEED)
    paths = [C.OUT / f"snapshot_pilot_E-G_s{T.SEED}_u{u:03d}.pt" for u in UPDATES]
    if not all(path.exists() for path in paths):
        raise FileNotFoundError("All four E-G pre-update snapshots are required")
    data_sources = {"items": T31.OUT / "items_warm.pt", "features": T31.OUT / "features_warm.pt"}
    data_hashes = {name: sha256_file(path) for name, path in data_sources.items()}
    items = torch.load(data_sources["items"], map_location="cpu", weights_only=False)[:8]
    feats = torch.load(data_sources["features"], map_location="cpu", weights_only=False)[:8]
    if len(items) != 8 or len(feats) != 8:
        raise ValueError("Eight fixed warm documents required")
    head = T.load_head()
    model, modules, params = T.init_model()
    refs = T.reference_logps(model, items, feats, head, check=guard)
    report = C.record("gradient_and_cloned_adamw_diagnostics", identities=identities,
        data_source_sha256=data_hashes, document_indices=list(range(8)),
        batches=[[2 * b, 2 * b + 1] for b in range(4)],
        diagnostic_seed=DIAGNOSTIC_SEED, snapshots=[], cap_seconds=cap,
        interpretation="local fixed-history and fixed-probability surrogate; observed online effects are evaluated separately")
    fixed_data = {"identities": identities, "data_source_sha256": data_hashes,
                  "reference_logp": refs, "snapshots": []}
    for expected_update, path in zip(UPDATES, paths):
        guard()
        snapshot = torch.load(path, map_location="cpu", weights_only=False)
        if snapshot["arm"] != "E-G" or snapshot["update"] != expected_update:
            raise ValueError("Snapshot arm or update mismatch")
        if snapshot.get("identities") != identities:
            raise ValueError("Snapshot inputs differ from the frozen protocol")
        steps = [float(state["step"]) for state in snapshot["opt"]["state"].values()]
        if any(step != expected_update for step in steps) or (expected_update > 0 and not steps):
            raise ValueError("Snapshot AdamW step counters mismatch")
        restore_parameters(params, snapshot["lora"])
        batches, all_contexts = [], []
        aggregate = {name: [torch.zeros_like(p, device="cpu") for p in params]
                     for name in COMPONENTS}
        mean_training_gen = 0.0
        for batch in range(4):
            indices = [2 * batch, 2 * batch + 1]
            grads, contexts, losses = batch_gradients(model, params, items, feats, refs,
                                                       head, indices, guard)
            summary, pooled = gradient_summary(grads)
            result = {"batch": batch, "document_indices": indices,
                      "losses": losses, "gradients": summary}
            result["adamw"] = counterfactual_steps(model, params, snapshot, pooled,
                items, feats, contexts, losses["gen"], guard)
            batches.append(result)
            all_contexts.extend(contexts)
            mean_training_gen += losses["gen"] / 4
            for name in COMPONENTS:
                for accumulator, value in zip(aggregate[name], grads[name]):
                    accumulator.add_(value, alpha=0.25)
            print(json.dumps({"update": expected_update, "batch": batch,
                              "control_gradient": summary["components"]["ctrl"],
                              "joint_minus_G_change": result["adamw"]["joint_minus_generation_only_loss_change"]}), flush=True)
            del grads, pooled
        summary, pooled = gradient_summary(aggregate)
        combined = {"document_indices": list(range(8)), "gradients": summary,
                    "adamw": counterfactual_steps(model, params, snapshot, pooled,
                        items, feats, all_contexts, mean_training_gen, guard)}
        report["snapshots"].append({"update": expected_update, "path": str(path),
            "sha256": sha256_file(path), "optimizer_step_counts": steps,
            "batches": batches, "eight_document_mean": combined})
        fixed_data["snapshots"].append({"update": expected_update, "contexts": all_contexts})
        report["seconds"] = time.perf_counter() - started
        report["completed"] = len(report["snapshots"]) == len(UPDATES)
        C.write_json(report_path, report)
        torch.save(fixed_data, data_path)
        print(json.dumps({"update": expected_update, "aggregate": summary["components"],
                          "seconds": report["seconds"]}), flush=True)
        del snapshot, aggregate, pooled
    report["fixed_inputs_path"] = str(data_path)
    report["fixed_inputs_sha256"] = sha256_file(data_path)
    C.write_json(report_path, report)
    print(json.dumps({"report": str(report_path), "completed": True}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
