"""Protocol section 5.2-5.4: deterministic manifest of documents, windows and roles.

Outputs (artifacts, not committed):
  manifest.jsonl  one line per selected document
  windows.pt      int32 [N, 256] token windows in manifest order
  rejects.jsonl   rows rejected because a special id appears inside the window
Receipt (committed): results/shared-depth-v2/data_manifest.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import struct
import unicodedata

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from .common import ARTIFACTS, MODEL_DIR, PARQUET, RESULTS, Timer, load_config, receipt, sha256_file, write_json

BOS = 65504
WINDOW = 256
SPECIAL_MIN = 65504


def normalize(text: str) -> str:
    return " ".join(unicodedata.normalize("NFKC", text).split())


def bucket_of(norm: str, salt: str) -> int:
    digest = hashlib.sha256(salt.encode("utf-8") + b"\x00" + norm.encode("utf-8")).digest()
    return int.from_bytes(digest, "big") % 100


def window_hash(ids: list[int], salt: str) -> str:
    return hashlib.sha256(salt.encode("utf-8") + b"\x00" + struct.pack(f"<{len(ids)}H", *ids)).hexdigest()


def role_table(cfg: dict) -> tuple[dict[int, str], dict[str, int]]:
    by_bucket, quota = {}, {}
    for role, spec in cfg["data"]["splits"].items():
        lo, hi = spec["buckets"]
        for b in range(lo, hi + 1):
            by_bucket[b] = role
        quota[role] = spec.get("documents", spec.get("maximum"))
    return by_bucket, quota


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-parquet-hash", action="store_true")
    args = parser.parse_args()
    cfg = load_config()
    data = cfg["data"]
    costs: dict = {}
    out_dir = ARTIFACTS / "data"
    out_dir.mkdir(parents=True, exist_ok=True)

    with Timer(costs, "parquet_hash_seconds"):
        parquet_sha = None if args.skip_parquet_hash else sha256_file(PARQUET)
    if parquet_sha is not None and parquet_sha != data["sha256"]:
        raise SystemExit(f"parquet sha256 mismatch: {parquet_sha}")
    if PARQUET.stat().st_size != data["file_bytes"]:
        raise SystemExit("parquet size mismatch")

    tokenizer = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    by_bucket, quota = role_table(cfg)
    filled = {role: 0 for role in quota}
    split_salt, window_salt = data["split_hash"]["salt"], data["window_hash"]["salt"]
    first, last_excl = data["first_source_row"], data["last_source_row_exclusive"]

    seen_dedup, seen_window = set(), set()
    selected, windows, rejects = [], [], []
    counts = {"rows_read": 0, "too_short": 0, "special_in_window": 0, "dup_text": 0,
              "dup_window": 0, "role_full": 0, "eligible_unique": 0}
    pf = pq.ParquetFile(PARQUET)
    row0 = 0
    done = False
    with Timer(costs, "scan_seconds"):
        for rg in range(pf.num_row_groups):
            n_rows = pf.metadata.row_group(rg).num_rows
            if row0 + n_rows <= first:
                row0 += n_rows
                continue
            table = pf.read_row_group(rg, columns=["text", "id"])
            texts, ids = table.column("text").to_pylist(), table.column("id").to_pylist()
            lo = max(first - row0, 0)
            hi = min(last_excl - row0, n_rows)
            batch_text = texts[lo:hi]
            encoded = tokenizer(batch_text, add_special_tokens=False, split_special_tokens=True)["input_ids"]
            for offset, (text, toks) in enumerate(zip(batch_text, encoded)):
                row = row0 + lo + offset
                counts["rows_read"] += 1
                seq = [BOS] + toks
                if len(seq) < WINDOW:
                    counts["too_short"] += 1
                    continue
                win = seq[:WINDOW]
                if any(t >= SPECIAL_MIN for t in win[1:]):
                    counts["special_in_window"] += 1
                    rejects.append({"source_row": row, "doc_id": ids[lo + offset], "reason": "special_id_in_window"})
                    continue
                norm = normalize(text)
                dkey = hashlib.sha256(norm.encode("utf-8")).hexdigest()
                if dkey in seen_dedup:
                    counts["dup_text"] += 1
                    continue
                seen_dedup.add(dkey)
                whash = window_hash(win, window_salt)
                if whash in seen_window:
                    counts["dup_window"] += 1
                    continue
                seen_window.add(whash)
                counts["eligible_unique"] += 1
                bucket = bucket_of(norm, split_salt)
                role = by_bucket[bucket]
                if filled[role] >= quota[role]:
                    counts["role_full"] += 1
                    continue
                filled[role] += 1
                selected.append({
                    "source_row": row, "doc_id": ids[lo + offset],
                    "raw_text_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                    "dedup_key_sha256": dkey, "window_token_sha256": whash,
                    "token_count": len(seq), "bucket": bucket, "role": role,
                })
                windows.append(win)
                if all(filled[r] >= quota[r] for r in quota):
                    done = True
                    break
            row0 += n_rows
            if done or row0 >= last_excl:
                break

    if not done:
        short = {r: (filled[r], quota[r]) for r in quota if filled[r] < quota[r]}
        raise SystemExit(f"quota not met before row {last_excl}: {short}")
    hashes = [s["window_token_sha256"] for s in selected]
    if len(set(hashes)) != len(hashes):
        raise SystemExit("window hashes not disjoint")

    manifest_path = out_dir / "manifest.jsonl"
    manifest_path.write_text("".join(json.dumps(s, sort_keys=True) + "\n" for s in selected),
                             encoding="utf-8", newline="\n")
    (out_dir / "rejects.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rejects),
                                           encoding="utf-8", newline="\n")
    window_tensor = torch.tensor(windows, dtype=torch.int32)
    torch.save(window_tensor, out_dir / "windows.pt")

    role_rows = {r: [s["source_row"] for s in selected if s["role"] == r] for r in quota}
    payload = receipt(
        "data_manifest",
        parquet={"path": str(PARQUET), "sha256": parquet_sha or "skipped", "bytes": PARQUET.stat().st_size},
        tokenizer_sha256=sha256_file(MODEL_DIR / "tokenizer.json"),
        counts=counts, last_row_scanned=selected[-1]["source_row"],
        roles={r: {"documents": filled[r], "first_row": min(role_rows[r]), "last_row": max(role_rows[r])}
               for r in quota},
        manifest_sha256=sha256_file(manifest_path),
        windows_sha256=hashlib.sha256(window_tensor.numpy().astype("<i4").tobytes()).hexdigest(),
        rejects=len(rejects), costs=costs)
    write_json(RESULTS / "data_manifest.json", payload)
    print(json.dumps({"counts": counts, "roles": payload["roles"], "costs": costs}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
