"""Protocol jev-chunk-depth-v3.1 sections 3.1, 5.1, 5.2: GSM8K manifest, chunking and initial states.

Items keep the phase-0 item format so the online runner in chunk_depth_v3.phase0_online can be reused:
{"split", "row", "ids", "chunk_of_pos", "first_solution_pos", "n_chunks", "answer_chunk", "gold"}.
There is no answer line in v3.1, so answer_chunk = n_chunks (a chunk index that never occurs).
"""

from __future__ import annotations

import hashlib
import json
import math
import re

import pyarrow.parquet as pq
import torch

from experiments.shared_depth_v2.common import MODEL_DIR, ROOT, receipt, sha256_file, write_json

GSM8K_COMMIT = "740312add88f781978c0658806c59bc2815b9866"
DATA = ROOT / "artifacts" / "gsm8k"
OUT = ROOT / "artifacts" / "chunk-depth-v31"
RESULTS = ROOT / "results" / "chunk-depth-v31"
BOS = 65504
MAX_TOKENS = 320
ROLES = {"phase0_survey": (0, 9), "formal_train": (10, 64), "warm": (65, 69), "pilot_train": (70, 74),
         "pilot_cal": (75, 77), "pilot_screen": (78, 84), "formal_cal": (85, 89), "formal_screen": (90, 99)}
SIGMA, L_CDF, U_CDF, EMBED_SCALE = 0.008703882797784892, 0.0013498980316301035, 0.9986501019683699, 72.66360473632812


def bucket(question: str) -> int:
    return int.from_bytes(hashlib.sha256(b"jev-chunk-v3\x00" + question.encode("utf-8")).digest(), "big") % 100


def state_matrix(split: str, row: int, positions: int, replica: int = 0) -> torch.Tensor:
    out = []
    for t in range(positions):
        key = b"\x00".join([b"jev-chunk-v3-state", GSM8K_COMMIT.encode(), split.encode(), str(row).encode(),
                            str(t).encode(), str(replica).encode()])
        g = torch.Generator(device="cpu").manual_seed(int.from_bytes(hashlib.sha256(key).digest()[:8], "big"))
        x = torch.empty(5280, dtype=torch.float32).uniform_(2 * L_CDF - 1, 2 * U_CDF - 1, generator=g)
        x.erfinv_().mul_(SIGMA * math.sqrt(2.0)).clamp_(-3 * SIGMA, 3 * SIGMA)
        x.mul_(torch.tensor(EMBED_SCALE, dtype=torch.float32))
        out.append(x.to(torch.bfloat16))
    return torch.stack(out)


def gold_answer(answer: str) -> str:
    m = re.search(r"####\s*(.+)", answer)
    return m.group(1).strip().replace(",", "") if m else ""


def make_item(tokenizer, split: str, row: int, question: str, answer: str):
    lines = [l for l in re.sub(r"<<[^>]*>>", "", answer).strip().split("\n") if not l.startswith("####")]
    solution = "\n".join(lines).strip()
    prompt = f"Question: {question.strip()}\nAnswer:"
    text = prompt + " " + solution
    sol_start = len(prompt) + 1
    enc = tokenizer(text, add_special_tokens=False, split_special_tokens=True, return_offsets_mapping=True)
    ids = [BOS] + enc["input_ids"]
    if len(ids) > MAX_TOKENS:
        return None
    line_starts = [sol_start] + [sol_start + m.end() for m in re.finditer("\n", solution)]
    chunk = [0]
    for (_a, b) in enc["offset_mapping"]:
        last = b - 1
        chunk.append(0 if last < sol_start else sum(1 for s in line_starts if s <= last))
    present = sorted(set(chunk) - {0})
    if not present:
        return None
    remap = {c: i + 1 for i, c in enumerate(present)}          # drop empty lines, keep order
    chunk = [0 if c == 0 else remap[c] for c in chunk]
    n_lines = len(present)
    return {"split": split, "row": row, "ids": ids, "chunk_of_pos": chunk,
            "first_solution_pos": next(i for i, c in enumerate(chunk) if c > 0),
            "n_chunks": n_lines + 1, "answer_chunk": n_lines + 1, "gold": gold_answer(answer)}


def main() -> int:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    OUT.mkdir(parents=True, exist_ok=True)
    by_role = {r: [] for r in ROLES}
    by_role["confirm"] = []
    counts = {"train_rows": 0, "test_rows": 0, "too_long": 0}
    for split in ("train", "test"):
        rows = pq.read_table(DATA / f"{split}.parquet").to_pylist()
        counts[f"{split}_rows"] = len(rows)
        for r, rec in enumerate(rows):
            item = make_item(tok, split, r, rec["question"], rec["answer"])
            if item is None:
                counts["too_long"] += 1
                continue
            if split == "test":
                by_role["confirm"].append(item)
                continue
            b = bucket(rec["question"])
            role = next(name for name, (lo, hi) in ROLES.items() if lo <= b <= hi)
            by_role[role].append(item)
    sizes = {}
    for role, items in by_role.items():
        torch.save(items, OUT / f"items_{role}.pt")
        sizes[role] = {"documents": len(items),
                       "mean_tokens": sum(len(i["ids"]) for i in items) / max(1, len(items)),
                       "mean_lines": sum(i["n_chunks"] - 1 for i in items) / max(1, len(items))}
    write_json(RESULTS / "data_manifest.json", receipt(
        "chunk_depth_v3_1_data_manifest", gsm8k_commit=GSM8K_COMMIT,
        parquet_sha256={s: sha256_file(DATA / f"{s}.parquet") for s in ("train", "test")},
        tokenizer_sha256=sha256_file(MODEL_DIR / "tokenizer.json"), counts=counts, roles=sizes))
    print(json.dumps(sizes, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
