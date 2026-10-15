"""Protocol section 6.1: analytic FLOP ledger for one request (batch 1, 128-token prefix, 128 outputs).

Convention: multiply-add = 2 FLOPs; only matrix products are counted (linear layers, attention
score/value products, adapter, LM head, readout). Attention FLOPs follow the shapes of the
executed SDPA calls: a prefill of S queries over K keys costs 4*S*K*h, as torch's FLOP counter
records it. LoRA is merged for inference and adds nothing.
"""

from __future__ import annotations

H = 5280
M = 17920
V = 65536
PREFIX = 128
OUTPUTS = 128
ACTIONS = 4


def block(S: int, K: int) -> int:
    linear = 2 * S * (3 * H * H + H * H + H * 2 * M + M * H)
    attention = 4 * S * K * H
    return linear + attention


def adapter(S: int) -> int:
    return 2 * S * (2 * H) * H


def stack(S: int, K: int, depth: int) -> int:
    """Prelude + depth core iterations + coda for S queries attending over K keys."""
    return 2 * block(S, K) + depth * (adapter(S) + 4 * block(S, K)) + 2 * block(S, K)


def lm_head(n: int) -> int:
    return 2 * n * H * V


def native(depth: int) -> int:
    prefill = stack(PREFIX, PREFIX, depth) + lm_head(1)
    decode = sum(stack(1, t + 1, depth) + lm_head(1) for t in range(PREFIX, PREFIX + OUTPUTS - 1))
    return prefill + decode


def readout() -> int:
    return 2 * H * ACTIONS


def coda_prefix() -> int:
    return 2 * block(PREFIX, PREFIX)


def plan(depth: int) -> int:
    return native(depth) + readout() + (coda_prefix() if depth > 4 else 0)


def normalized(depths=(4, 8, 16, 32)) -> dict:
    base = native(16)
    return {
        "denominator_flops": base,
        "f_plan": {d: plan(d) / base for d in depths},
        "f_static": {d: native(d) / base for d in depths},
        "native_flops": {d: native(d) for d in depths},
        "plan_flops": {d: plan(d) for d in depths},
        "readout_flops": readout(),
        "coda_prefix_flops": coda_prefix(),
    }
