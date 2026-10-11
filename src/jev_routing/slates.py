"""Choose an entire feasible computation plan in one real Canopy question."""
from __future__ import annotations
from dataclasses import dataclass, replace
import json
from .schema import RoutingContext, RoutePlan, RouteDecision
from .backend import DecisionBackend, validate_answers


@dataclass(frozen=True)
class PlanSlate:
    id: str
    description: str
    decisions: tuple[RouteDecision, ...]


class PlanSlateRouter:
    """Joint token allocation, expert selection and depth through a plan menu.

    Menus are supplied by the caller, not generated for free. Causal mode only
    allows a single current target. Multi-token slates require an explicit
    bidirectional/prefix-computation problem, whose output follows that prefix.
    """
    def __init__(self, backend: DecisionBackend):
        self.backend = backend

    def route(self, context: RoutingContext, slates: tuple[PlanSlate, ...], *, budget: int) -> RoutePlan:
        if type(budget) is not int or budget < 0:
            raise ValueError("budget must be a nonnegative integer")
        if not 1 <= len(slates) <= 10 or len({s.id for s in slates}) != len(slates):
            raise ValueError("Supply 1..10 uniquely named complete plan candidates")
        if context.mode == "causal" and len(context.targets) != 1:
            raise ValueError("Multi-token complete plans require explicit bidirectional/prefix-computation mode")
        targets = {t.position for t in context.targets}
        experts = {e.id: e for e in context.experts}
        feasible = []
        for slate in slates:
            if not slate.id or not slate.description.strip():
                raise ValueError("Plans need an ID and execution description")
            positions = [d.position for d in slate.decisions]
            if len(set(positions)) != len(positions) or set(positions) != targets:
                raise ValueError("Each slate must specify exactly one action for every target, including skips")
            spent, calls = 0, {e: 0 for e in experts}
            for decision in slate.decisions:
                action = decision.action
                if action.expert_id is not None:
                    if action.expert_id not in experts:
                        raise ValueError("Slate references an unknown expert")
                    calls[action.expert_id] += action.steps
                spent += action.cost
            if spent <= budget and all(experts[e].capacity is None or calls[e] <= experts[e].capacity for e in experts):
                feasible.append((slate, spent, calls))
        if not feasible:
            raise ValueError("No supplied complete plan satisfies the hard budget and capacities")
        state = context.for_target(context.targets[-1])
        criteria = {}
        for slate, spent, calls in feasible:
            spec = [{"position": d.position, "expert_id": d.action.expert_id,
                     "additional_steps": d.action.steps, "cost": d.action.cost,
                     "action_contract": d.action.description} for d in slate.decisions]
            criteria[slate.id] = json.dumps({"plan": slate.description, "actions": spec,
                "declared_total_cost": spent, "expert_step_counts": calls}, ensure_ascii=False)
        question = {"id": "complete_plan", "type": "choice",
            "question": "Choose the complete feasible computation plan most likely to improve the stated task. Use expert capability evidence, dependencies, current depth and observations. The plan menu is exhaustive only over the supplied candidates; do not infer that probability is measured loss improvement.",
            "criteria": criteria}
        answer = validate_answers([question], self.backend.decide(state, [question]))[0]
        p = answer["probabilities"]
        best = max(feasible, key=lambda item: p[item[0].id])
        slate, spent, calls = best
        return RoutePlan(tuple(replace(d, probabilities=dict(p)) for d in slate.decisions),
            spent, calls, "complete-plan-choice", {"selected_slate": slate.id,
                "slate_probabilities": p, "supplied_slates": len(slates),
                "feasible_slates": len(feasible), "objective": "Canopy choice over complete supplied plans"})
