"""Serialize full legal context, never an implicit suffix or a silent summary."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
from .schema import RoutingContext, TokenState


def make_state(context: RoutingContext, target: TokenState | int) -> str:
    position = target.position if isinstance(target, TokenState) else target
    found = [t for t in context.targets if t.position == position]
    if len(found) != 1:
        raise ValueError("Target must belong to this routing context")
    cutoff = position if context.mode == "causal" else len(context.tokens) - 1
    legal_tokens = list(context.tokens[:cutoff + 1])
    states = [asdict(t) for t in context.targets if t.position <= cutoff]
    payload = {
        "schema": "jev-routing-context-v1",
        "mode": context.mode,
        "task": context.task,
        "visible_through": cutoff,
        "tokens": [{"position": i, "text": text} for i, text in enumerate(legal_tokens)],
        "target": asdict(found[0]),
        "visible_states": states,
        "experts": [asdict(e) for e in context.experts],
        "history": [entry for entry in context.history if entry["visible_after"] <= cutoff],
        "metadata": [entry for entry in context.metadata if entry["visible_after"] <= cutoff],
        "contract": "Observations are supplied by the host. Capability cards and task instructions must be fixed before the current prediction. No future labels or target answers may appear here.",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def context_digest(state: str) -> str:
    return hashlib.sha256(state.encode("utf-8")).hexdigest()
