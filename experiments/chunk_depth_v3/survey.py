"""Phase 0 of protocol jev-chunk-depth-v3: depth headroom survey on GSM8K with the frozen model.

S1  fixed-depth tables: per-token NLL of the reference solution at depths 4..64 (no cache).
S2  mixed-depth check: chunk-wise prefill with Huginn's latest-m4 KV lookup and random
    per-chunk depths, compared with the fixed-depth table entries for the same chunks.
Chunks: the question prompt is chunk 0; each solution line is one chunk. A chunk owns the
positions of its tokens, and its scored tokens are those predicted from its positions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import time

import pyarrow.parquet as pq
import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import MODEL_DIR, ROOT, receipt, sha256_file, write_json

GSM8K_COMMIT = "740312add88f781978c0658806c59bc2815b9866"
DATA = ROOT / "artifacts" / "gsm8k"
OUT = ROOT / "artifacts" / "chunk-depth-v3"
RESULTS = ROOT / "results" / "chunk-depth-v3"
DEPTHS = (4, 6, 8, 12, 16, 24, 32, 48, 64)
MAX_TOKENS = 320
BOS = 65504
SIGMA, L_CDF, U_CDF, EMBED_SCALE = 0.008703882797784892, 0.0013498980316301035, 0.9986501019683699, 72.66360473632812


def bucket(question: str) -> int:
    return int.from_bytes(hashlib.sha256(b"jev-chunk-v3\x00" + question.encode("utf-8")).digest(), "big") % 100


def state_matrix(row: int, positions: int, replica: int = 0) -> torch.Tensor:
    out = []
    for t in range(positions):
        key = b"\x00".join([b"jev-chunk-v3-state", GSM8K_COMMIT.encode(), b"train", str(row).encode(),
                            str(t).encode(), str(replica).encode()])
        g = torch.Generator(device="cpu").manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
        x = torch.empty(5280, dtype=torch.float32).uniform_(2 * L_CDF - 1, 2 * U_CDF - 1, generator=g)
        x.erfinv_().mul_(SIGMA * math.sqrt(2.0)).clamp_(-3 * SIGMA, 3 * SIGMA)
        x.mul_(torch.tensor(EMBED_SCALE, dtype=torch.float32))
        out.append(x.to(torch.bfloat16))
    return torch.stack(out)


def build_items(tokenizer, buckets: range):
    table = pq.read_table(DATA / "train.parquet").to_pylist()
    items = []
    for row, rec in enumerate(table):
        if bucket(rec["question"]) not in buckets:
            continue
        solution = re.sub(r"<<[^>]*>>", "", rec["answer"]).strip()
        prompt = f"Question: {rec['question'].strip()}\nAnswer:"
        text = prompt + " " + solution
        sol_start = len(prompt) + 1
        enc = tokenizer(text, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
        ids = [BOS] + enc["input_ids"]
        if len(ids) > MAX_TOKENS:
            continue
        line_starts = [sol_start] + [sol_start + m.end() for m in re.finditer("\n", solution)]
        chunk_of_pos = [0]
        for (_a, b) in enc["offset_mapping"]:
            last = b - 1
            if last < sol_start:
                chunk_of_pos.append(0)
            else:
                chunk_of_pos.append(sum(1 for s in line_starts if s <= last))
        first_sol = next(i for i, c in enumerate(chunk_of_pos) if c > 0)
        answer_line = len(line_starts)
        items.append({"row": row, "ids": ids, "chunk_of_pos": chunk_of_pos, "first_solution_pos": first_sol,
                      "n_chunks": answer_line + 1, "answer_chunk": answer_line})
    return items


@torch.no_grad()
def fixed_tables(model, item, device):
    ids = torch.tensor(item["ids"], device=device)[None]
    S = ids.shape[1]
    fc = H.freqs(model, torch.arange(S))
    state = state_matrix(item["row"], S).to(device)[None]
    e = H.prelude(model, ids, fc)
    r, done, rows = state, 0, []
    targets = ids[0, 1:]
    for d in DEPTHS:
        r = H.core_steps(model, r, e, fc, done, d)
        done = d
        logits = H.head(model, H.coda(model, H.ln_f(model, r), fc))[0, :-1]
        rows.append(torch.nn.functional.cross_entropy(logits, targets, reduction="none").float().cpu())
    return torch.stack(rows)  # [depths, S-1]; column t is NLL of token t+1 predicted from position t


@torch.no_grad()
def mixed_run(model, item, depth_of_chunk: list[int], device):
    """Chunk-wise prefill with latest-m4 KV lookup; returns per-position NLL [S-1]."""
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("raven_modeling_minimal.HuginnDynamicCache", str(MODEL_DIR))
    cache = cls(lookup_strategy="latest-m4")
    ids = torch.tensor(item["ids"], device=device)[None]
    S = ids.shape[1]
    state = state_matrix(item["row"], S).to(device)[None]
    chunk = torch.tensor(item["chunk_of_pos"])
    out = torch.empty(S - 1)
    a = 0
    while a < S:
        c = int(chunk[a])
        b = a
        while b < S and int(chunk[b]) == c:
            b += 1
        fc = H.freqs(model, torch.arange(a, b))
        e = H.prelude(model, ids[:, a:b], fc, cache)
        r = H.core_steps(model, state[:, a:b], e, fc, 0, depth_of_chunk[c], cache)
        logits = H.head(model, H.coda(model, H.ln_f(model, r), fc, cache))[0]
        hi = min(b, S - 1)
        if hi > a:
            out[a:hi] = torch.nn.functional.cross_entropy(logits[: hi - a], ids[0, a + 1:hi + 1],
                                                          reduction="none").float().cpu()
        a = b
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mixed-items", type=int, default=48)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    from transformers import AutoTokenizer

    costs = {}
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    items = build_items(tok, range(0, 10))
    if args.limit:
        items = items[: args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    model = H.load_model()
    costs["model_load_seconds"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    tables = []
    for k, item in enumerate(items):
        tables.append(fixed_tables(model, item, "cuda"))
        if (k + 1) % 100 == 0:
            print(f"S1 {k + 1}/{len(items)} {time.perf_counter() - t0:.0f}s", flush=True)
    costs["s1_seconds"] = time.perf_counter() - t0
    torch.save({"items": items, "tables": tables, "depths": DEPTHS}, OUT / "survey_tables.pt")

    g = torch.Generator(device="cpu").manual_seed(20261012)
    choices = (4, 8, 16, 32)
    mixed = []
    t0 = time.perf_counter()
    for k, item in enumerate(items[: args.mixed_items]):
        assign = [choices[int(i)] for i in torch.randint(0, 4, (item["n_chunks"],), generator=g)]
        uniform = [16] * item["n_chunks"]
        m = mixed_run(model, item, assign, "cuda")
        u = mixed_run(model, item, uniform, "cuda")
        mixed.append({"index": k, "assign": assign, "mixed_nll": m, "uniform16_nll": u})
        if (k + 1) % 8 == 0:
            print(f"S2 {k + 1}/{args.mixed_items} {time.perf_counter() - t0:.0f}s", flush=True)
    costs["s2_seconds"] = time.perf_counter() - t0
    torch.save({"mixed": mixed, "choices": choices}, OUT / "survey_mixed.pt")
    payload = receipt("chunk_depth_v3_phase0_survey", data={"gsm8k_commit": GSM8K_COMMIT,
                      "train_parquet_sha256": sha256_file(DATA / "train.parquet"), "survey_buckets": [0, 9],
                      "items": len(items), "max_tokens": MAX_TOKENS}, depths=DEPTHS, costs=costs,
                      mixed_items=len(mixed))
    write_json(RESULTS / "survey_run.json", payload)
    print(json.dumps({"items": len(items), "costs": costs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
