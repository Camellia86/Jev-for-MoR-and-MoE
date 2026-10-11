"""Run a controlled, hand-authored routing replay without loading Canopy-Jev.

The input fixture is a protocol test: its probabilities exercise context,
candidate-axis, cost, and capacity contracts.  It does not evaluate a trained
router, establish task accuracy, or substitute for actual Canopy-Jev output.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_routing.backend import ReplayBackend
from jev_routing.routers import JointPlanRouter
from jev_routing.schema import Action, Expert, RoutingContext, TokenState


def load_fixture(path: Path) -> tuple[RoutingContext, dict[int, list[Action]], int, dict[str, dict[str, float]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "fixture_notice" not in data:
        raise ValueError("Replay inputs require an explicit fixture_notice")
    raw = data["routing_context"]
    context = RoutingContext(
        tokens=tuple(raw["tokens"]),
        targets=tuple(TokenState(**item) for item in raw["targets"]),
        experts=tuple(Expert(**item) for item in raw["experts"]),
        task=raw["task"], mode=raw.get("mode", "causal"),
        history=tuple(raw.get("history", [])), metadata=tuple(raw.get("metadata", [])),
    )
    actions = {item["id"]: Action(**item) for item in data["actions"]}
    offers = {int(position): [actions[action_id] for action_id in action_ids]
              for position, action_ids in data["offers"].items()}
    return context, offers, int(data["budget"]), data["replay_probabilities"]


def serialise_plan(plan) -> dict:
    return {
        "decisions": [
            {"position": decision.position, "action": decision.action.id,
             "expert": decision.action.expert_id, "steps": decision.action.steps,
             "cost": decision.action.cost, "routing_proxy_score": decision.score,
             "fixture_probabilities": dict(decision.probabilities)}
            for decision in plan.decisions
        ],
        "spent": plan.spent,
        "expert_steps": dict(plan.expert_calls),
        "algorithm": plan.algorithm,
        "diagnostics": dict(plan.diagnostics),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path,
                        default=ROOT / "examples" / "request.json")
    args = parser.parse_args()
    context, offers, budget, probabilities = load_fixture(args.request)
    backend = ReplayBackend(probabilities)
    plan = JointPlanRouter(backend, budget=budget).route(context, offers)
    print(json.dumps({
        "result_kind": "controlled-fixture-replay",
        "notice": "Hand-authored probabilities validate protocol execution only; no Canopy-Jev inference or training metric was run.",
        "result": serialise_plan(plan),
        "backend_requests": len(backend.requests),
    }, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
