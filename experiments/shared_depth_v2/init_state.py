"""Protocol section 5.5: per-position initial recurrent state, independent of text content."""

from __future__ import annotations

import hashlib
import math

import torch

HIDDEN = 5280
DATASET_COMMIT = "3ba9d605774198c5868892d7a8deda78031a781f"
KEY_PREFIX = "jev-shared-v2-state"
L_CDF = 0.0013498980316301035
U_CDF = 0.9986501019683699
SIGMA = 0.008703882797784892
EMBED_SCALE_FP32 = 72.66360473632812


def seed_for(row: int, position: int, replica: int, commit: str = DATASET_COMMIT) -> int:
    key = b"\x00".join([KEY_PREFIX.encode(), commit.encode(), str(row).encode(),
                        str(position).encode(), str(replica).encode()])
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big")


def state_vector(row: int, position: int, replica: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu").manual_seed(seed_for(row, position, replica))
    x = torch.empty(HIDDEN, dtype=torch.float32)
    x.uniform_(2 * L_CDF - 1, 2 * U_CDF - 1, generator=generator)
    x.erfinv_()
    x.mul_(SIGMA * math.sqrt(2.0))
    x.clamp_(-3 * SIGMA, 3 * SIGMA)
    x.mul_(torch.tensor(EMBED_SCALE_FP32, dtype=torch.float32))
    return x.to(torch.bfloat16)


def state_matrix(row: int, positions: int, replica: int) -> torch.Tensor:
    """BF16 [positions, 5280] initial state for absolute positions 0..positions-1."""
    return torch.stack([state_vector(row, t, replica) for t in range(positions)])


def bf16_le_bytes(x: torch.Tensor) -> bytes:
    return x.contiguous().view(torch.int16).numpy().astype("<i2").tobytes()


def verify_test_vectors(vectors: list[dict]) -> list[dict]:
    rows = []
    for vec in vectors:
        seed = seed_for(vec["source_row"], vec["position"], vec["replica"])
        x = state_vector(vec["source_row"], vec["position"], vec["replica"])
        raw = bf16_le_bytes(x)
        rows.append({
            **{k: vec[k] for k in ("source_row", "position", "replica")},
            "seed_match": str(seed) == vec["seed64"],
            "first8_match": raw[:16].hex() == vec["first_8_bf16_le_hex"],
            "sha256_match": hashlib.sha256(raw).hexdigest() == vec["sha256_bf16_le"],
            "observed_sha256": hashlib.sha256(raw).hexdigest(),
        })
    return rows
