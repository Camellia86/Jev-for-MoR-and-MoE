"""Route the example with the actual Canopy-Jev model, never a replay fixture.

This command needs Canopy-Jev at source commit
b71f97b3bffd28c90306bfea607d2089d75e7d43, its artifacts and base checkpoint,
and an NVIDIA GPU with the memory required by Canopy-Jev.  It is deliberately
not a CPU fallback or a claim that the included replay has run this model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_routing.backend import BASE_REVISION, CANOPY_REVISION, CanopyBackend
from jev_routing.routers import JointPlanRouter
from jev_routing.schema import Action, Expert, RoutingContext, TokenState


def load_request(path: Path) -> tuple[RoutingContext, dict[str, list[Action]], int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload["routing_context"]
    context = RoutingContext(
        tokens=tuple(raw["tokens"]),
        targets=tuple(TokenState(**target) for target in raw["targets"]),
        experts=tuple(Expert(**expert) for expert in raw["experts"]),
        task=raw["task"],
        mode=raw.get("mode", "causal"),
        history=tuple(raw.get("history", [])),
        metadata=tuple(raw.get("metadata", [])),
    )
    actions = {item["id"]: Action(**item) for item in payload["actions"]}
    offers = {int(position): [actions[action_id] for action_id in ids]
              for position, ids in payload["offers"].items()}
    return context, offers, int(payload["budget"])


def as_jsonable(plan) -> dict:
    return {
        "decisions": [
            {"position": decision.position, "action": decision.action.id,
             "expert": decision.action.expert_id, "steps": decision.action.steps,
             "cost": decision.action.cost, "score": decision.score,
             "probabilities": dict(decision.probabilities)}
            for decision in plan.decisions
        ],
        "spent": plan.spent,
        "expert_steps": dict(plan.expert_calls),
        "algorithm": plan.algorithm,
        "diagnostics": dict(plan.diagnostics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", required=True, help="Canopy-Jev artifact directory")
    parser.add_argument("--base-model", required=True, help="Exact base-model directory or repository")
    parser.add_argument("--revision", default=BASE_REVISION,
                        help="Pinned base revision (override only for a verified compatible base)")
    parser.add_argument("--request", type=Path, default=Path(__file__).with_name("request.json"))
    args = parser.parse_args()

    context, offers, budget = load_request(args.request)
    backend = CanopyBackend.from_pretrained(
        args.artifacts, base_model=args.base_model, revision=args.revision,
        local_files_only=False,
    )
    plan = JointPlanRouter(backend, budget=budget).route(context, offers)
    print(json.dumps({
        "backend": "actual-canopy-jev",
        "canopy_source_commit": CANOPY_REVISION,
        "result": as_jsonable(plan),
        "receipts": backend.receipts,
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
