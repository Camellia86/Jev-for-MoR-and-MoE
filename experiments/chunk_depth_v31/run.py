"""Online execution under the v3.1 semantics (sections 3.2-3.4) and shared helpers."""

from __future__ import annotations

import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import MODEL_DIR
from experiments.chunk_depth_v3.analyze_survey import position_cost
from experiments.chunk_depth_v31.data import state_matrix

V3 = (4, 6, 8, 12, 16)
R_REL = torch.tensor([0.301, 0.418, 0.534, 0.767, 1.000], dtype=torch.float64)


def lines_of(item):
    S, chunk = len(item["ids"]), item["chunk_of_pos"]
    lines = []
    for c in range(1, item["n_chunks"]):
        pos = [t for t in range(S) if chunk[t] == c]
        lines.append((pos[0], pos[-1] + 1))
    return lines[0][0], lines, S


def scored_in(item, a, b):
    S = len(item["ids"])
    return [t for t in range(a, b) if t <= S - 2]


def pos_feats(c, prev_len):
    return [1.0 if min(c, 9) == i else 0.0 for i in range(1, 10)] + [prev_len / 50.0]


def line_cost(a, b, d):
    return sum(position_cost(t, d) for t in range(a, b))


def new_cache():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module
    cls = get_class_from_dynamic_module("raven_modeling_minimal.HuginnDynamicCache", str(MODEL_DIR))
    return cls(lookup_strategy="latest-m4")


def item_state(item, S):
    return state_matrix(item["split"], item["row"], S).cuda()[None]


@torch.no_grad()
def run_segment(model, cache, ids, state, a, b, d):
    fc = H.freqs(model, torch.arange(a, b))
    e = H.prelude(model, ids[:, a:b], fc, cache)
    r = H.core_steps(model, state[:, a:b], e, fc, 0, d, cache)
    z = H.coda(model, H.ln_f(model, r), fc, cache)
    return z[0], H.head(model, z)[0]


@torch.no_grad()
def online(model, item, policy, q_depth=16, return_reps=False):
    """Question at q_depth, then each line's depth = policy(z, depth_of_z, line_index, prev_len, history)
    where z is the actual coda output at the last position of the previous segment."""
    q_end, lines, S = lines_of(item)
    ids = torch.tensor(item["ids"], device="cuda")[None]
    state = item_state(item, S)
    cache = new_cache()
    nll = torch.full((S - 1,), float("nan"))

    def record(a, b, logits):
        hi = min(b, S - 1)
        if hi > a:
            nll[a:hi] = torch.nn.functional.cross_entropy(logits[: hi - a].float(), ids[0, a + 1:hi + 1],
                                                          reduction="none").cpu()

    z, logits = run_segment(model, cache, ids, state, 0, q_end, q_depth)
    record(0, q_end, logits)
    last_z, last_d, prev_len = z[-1].float().cpu(), q_depth, q_end
    depths, history, reps = [], [], []
    for c, (a, b) in enumerate(lines, start=1):
        d = policy(last_z, last_d, c, prev_len, history)
        reps.append(last_z)
        z, logits = run_segment(model, cache, ids, state, a, b, d)
        record(a, b, logits)
        depths.append(d)
        history.append((b - a, d))
        last_z, last_d, prev_len = z[-1].float().cpu(), d, b - a
    return (nll, depths, reps) if return_reps else (nll, depths)


def doc_stats(item, nll, depths):
    _, lines, _ = lines_of(item)
    sc = [t for (a, b) in lines for t in scored_in(item, a, b)]
    vals = nll[sc].double()
    return {"mean_nll": float(vals.mean()), "sum_nll": float(vals.sum()), "tokens": len(sc),
            "cost": sum(line_cost(a, b, d) for (a, b), d in zip(lines, depths)),
            "base16": sum(line_cost(a, b, 16) for (a, b) in lines), "depths": list(depths)}


def retry(fn, *args, **kwargs):
    for attempt in range(6):
        try:
            return fn(*args, **kwargs)
        except (torch.OutOfMemoryError, RuntimeError) as exc:
            if "out of memory" not in str(exc).lower():
                raise
            torch.cuda.empty_cache()
            time.sleep(10 * (attempt + 1))
    raise RuntimeError("repeated OOM")


def coverage(history, d):
    """Shortfall s and excess e of plan section 2.5 over previous solution lines."""
    if not history:
        return 0.0, 0.0
    m = sum(n for n, _ in history)
    s = sum(n * max(d - dj, 0) for n, dj in history) / m
    e = sum(n * max(dj - d, 0) for n, dj in history) / m
    return s, e


def bootstrap(diff, repeats=10000, seed=20261013):
    x = torch.tensor(diff, dtype=torch.float64)
    g = torch.Generator().manual_seed(seed)
    m = x[torch.randint(0, len(x), (repeats, len(x)), generator=g)].mean(1)
    return {"mean": float(x.mean()), "sd": float(x.std()), "n": len(x),
            "ci95": [float(torch.quantile(m, 0.025)), float(torch.quantile(m, 0.975))],
            "lower_one_sided_90": float(torch.quantile(m, 0.10)),
            "lower_one_sided_95": float(torch.quantile(m, 0.05))}
