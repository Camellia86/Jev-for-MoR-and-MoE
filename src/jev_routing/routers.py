"""Jev-driven routing policies with explicit score semantics.

Each policy serializes a target's complete legal context through
``RoutingContext.for_target`` before asking the Canopy backend.  The backend
may batch only questions with byte-identical serialized states; prefixes are
never reconstructed by the router and no future token is appended locally.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, replace
import math
from typing import Any, Callable, Iterable, Mapping, Sequence, Union

from .allocation import allocate
from .backend import DecisionBackend
from .schema import (
    Action,
    AllocationOption,
    Expert,
    RouteDecision,
    RoutePlan,
    RoutingContext,
    TokenState,
)


ActionOffers = Union[Mapping[int, Iterable[Action]], Iterable[Action]]
PerPassOffers = Union[ActionOffers, Callable[[RoutingContext, int], ActionOffers]]


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _normalise_actions(
    context: RoutingContext,
    offers: ActionOffers,
    *,
    enforce_choice_limit: bool = True,
) -> dict[int, tuple[Action, ...]]:
    if isinstance(offers, Mapping):
        raw = {target.position: tuple(offers.get(target.position, ())) for target in context.targets}
        unknown = set(offers) - set(raw)
        if unknown:
            raise ValueError(f"Action offers include targets not in context: {sorted(unknown)}")
    else:
        shared = tuple(offers)
        raw = {target.position: shared for target in context.targets}
    known = {expert.id for expert in context.experts}
    result: dict[int, tuple[Action, ...]] = {}
    for position, actions in raw.items():
        seen: set[str] = set()
        skips = 0
        canonical: list[Action] = []
        for action in actions:
            if not isinstance(action, Action):
                raise ValueError("Action offers must contain Action values")
            if action.id in seen:
                raise ValueError(f"Duplicate action ID {action.id!r} at position {position}")
            if action.expert_id is not None and action.expert_id not in known:
                raise ValueError(f"Action {action.id!r} names an unknown expert")
            seen.add(action.id)
            if action.expert_id is None:
                skips += 1
            canonical.append(action)
        if skips > 1:
            raise ValueError(f"Position {position} offers more than one skip action")
        if skips == 0:
            canonical.append(Action.skip())
        if enforce_choice_limit and len(canonical) > 10:
            raise ValueError(
                f"Position {position} has {len(canonical)} route candidates; "
                "Canopy Choice accepts at most 10 including skip"
            )
        result[position] = tuple(canonical)
    return result


def _action_text(action: Action) -> str:
    if action.expert_id is None:
        return action.description or "Do no additional expert work."
    return (
        f"{action.description or 'Apply the selected expert.'} "
        f"Expert={action.expert_id}; recursive steps={action.steps}; declared cost={action.cost}."
    )


def _decide_grouped(
    backend: DecisionBackend,
    queries: Iterable[tuple[int, str, dict]],
) -> dict[str, dict]:
    """Ask shared states together and return the normalized answers by ID."""
    query_list = tuple(queries)
    grouped: dict[str, list[dict]] = defaultdict(list)
    for _, state, question in query_list:
        grouped[state].append(question)
    answers: dict[str, dict] = {}
    for state, questions in grouped.items():
        result = backend.decide(state, questions)
        if len(result) != len(questions):
            raise ValueError("Decision backend returned the wrong number of answers")
        for answer in result:
            identifier = str(answer.get("id", ""))
            if identifier in answers:
                raise ValueError(f"Decision backend duplicated answer ID {identifier!r}")
            answers[identifier] = answer
    expected = {str(question["id"]) for _, _, question in query_list}
    if set(answers) != expected:
        raise ValueError("Decision backend returned an unexpected answer axis")
    return answers


def _probabilities(answer: dict, expected: Iterable[str]) -> dict[str, float]:
    probabilities = answer.get("probabilities")
    if not isinstance(probabilities, Mapping) or set(probabilities) != set(expected):
        raise ValueError("Decision backend returned the wrong probability candidates")
    result = {str(key): float(value) for key, value in probabilities.items()}
    if not all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in result.values()):
        raise ValueError("Decision backend returned invalid probabilities")
    if not math.isclose(sum(result.values()), 1.0, abs_tol=2e-5):
        raise ValueError("Decision backend probabilities do not sum to one")
    return result


def _attach_probabilities(
    plan: RoutePlan,
    probabilities: Mapping[tuple[int, str], Mapping[str, float]],
    *,
    algorithm: str,
    diagnostics: Mapping[str, Any],
) -> RoutePlan:
    merged = dict(plan.diagnostics)
    merged.update(diagnostics)
    return RoutePlan(
        decisions=tuple(
            replace(
                decision,
                probabilities=dict(probabilities.get((decision.position, decision.action.id), {})),
            )
            for decision in plan.decisions
        ),
        spent=plan.spent,
        expert_calls=plan.expert_calls,
        algorithm=algorithm,
        diagnostics=merged,
    )


def _residual_experts_from(
    experts: Iterable[Expert], used_steps: Mapping[str, int]
) -> tuple[Expert, ...]:
    """Expose actual remaining step capacity to both JEV and the allocator."""
    result = []
    for expert in experts:
        if expert.capacity is None:
            result.append(expert)
            continue
        remaining = expert.capacity - int(used_steps.get(expert.id, 0))
        if remaining < 0:
            raise ValueError("Execution already exceeded an expert capacity")
        result.append(Expert(expert.id, expert.description, remaining))
    return tuple(result)


def _state_with_resources(
    context: RoutingContext,
    target: TokenState,
    *,
    remaining_budget: int,
    experts: Iterable[Expert],
) -> str:
    """Attach ex-ante resource state at this causal position before scoring."""
    expert_list = tuple(experts)
    record = {
        "visible_after": target.position,
        "kind": "routing_resources",
        "at_position": target.position,
        "remaining_budget": remaining_budget,
        "budget_unit": "Action.cost",
        "remaining_expert_step_capacity": {
            expert.id: expert.capacity for expert in expert_list
        },
    }
    enriched = replace(
        context,
        experts=expert_list,
        metadata=(*context.metadata, record),
    )
    return enriched.for_target(target)


class JointPlanRouter:
    """Choose one joint ``expert × depth`` action with a Choice question.

    ``score`` in the returned plan is a relative MAP preference
    ``log p(action) - log p(skip)``.  It is a routing proxy only and must not be
    read as expected NLL improvement or as a task reward.
    """

    def __init__(
        self,
        backend: DecisionBackend,
        *,
        budget: int | None = None,
        max_states: int = 100_000,
        probability_floor: float = 1e-12,
    ):
        if budget is not None:
            _integer(budget, "budget")
        _integer(max_states, "max_states", 1)
        if not isinstance(probability_floor, (int, float)) or not 0.0 < probability_floor <= 1.0:
            raise ValueError("probability_floor must be in (0, 1]")
        self.backend = backend
        self.budget = budget
        self.max_states = max_states
        self.probability_floor = float(probability_floor)

    def _budget(self, budget: int | None) -> int:
        resolved = self.budget if budget is None else budget
        if resolved is None:
            raise ValueError("Provide a routing budget in Action.cost units")
        _integer(resolved, "budget")
        return resolved

    def _route_causal(
        self,
        context: RoutingContext,
        offered: Mapping[int, tuple[Action, ...]],
        *,
        budget: int,
        experts: Iterable[Expert],
        target_positions: Iterable[int] | None = None,
    ) -> RoutePlan:
        """Commit each causal target before even scoring the next one.

        A global knapsack over scores from later target positions would make an
        earlier route change when a suffix changes, even if the earlier JEV
        state is a legal prefix.  Here the initial resource budget is assumed
        to be ex-ante metadata and every local decision consumes the remaining
        resource state immediately.
        """
        base_experts = tuple(experts)
        selected_positions = (
            {target.position for target in context.targets}
            if target_positions is None
            else set(target_positions)
        )
        targets = [
            target for target in sorted(context.targets, key=lambda item: item.position)
            if target.position in selected_positions
        ]
        if not targets:
            raise ValueError("No requested causal targets belong to this context")
        remaining_budget = budget
        used_steps = {expert.id: 0 for expert in base_experts}
        decisions: list[RouteDecision] = []
        commitment_trace: list[dict[str, Any]] = []
        for target in targets:
            action_list = offered[target.position]
            question = {
                "id": f"joint:{target.position}",
                "type": "choice",
                "instructions": (
                    "Select exactly one next routing action for this target. "
                    "The listed costs are hard resource declarations; choose based on "
                    "the legal state and the action descriptions."
                ),
                "criteria": {action.id: _action_text(action) for action in action_list},
            }
            answer = _decide_grouped(
                self.backend,
                [(
                    target.position,
                    _state_with_resources(
                        context,
                        target,
                        remaining_budget=remaining_budget,
                        experts=_residual_experts_from(base_experts, used_steps),
                    ),
                    question,
                )],
            )[str(question["id"])]
            probabilities = _probabilities(answer, (action.id for action in action_list))
            skip = next(action for action in action_list if action.expert_id is None)
            log_skip = math.log(max(probabilities[skip.id], self.probability_floor))
            local_options = [
                AllocationOption(target.position, skip, 0.0)
            ]
            for action in action_list:
                if action.expert_id is not None:
                    local_options.append(AllocationOption(
                        target.position,
                        action,
                        math.log(max(probabilities[action.id], self.probability_floor)) - log_skip,
                    ))
            local_plan = allocate(
                local_options,
                _residual_experts_from(base_experts, used_steps),
                budget=remaining_budget,
                max_states=self.max_states,
                algorithm="causal-online-relative-preference",
                diagnostics={
                    "score_semantics": (
                        "log(max(p(action), eps)) - log(max(p(skip), eps)); "
                        "a relative MAP routing proxy, not task utility"
                    ),
                },
            )
            decision = local_plan.decisions[0]
            decisions.append(replace(decision, probabilities=dict(probabilities)))
            remaining_budget -= local_plan.spent
            for expert_id, count in local_plan.expert_calls.items():
                used_steps[expert_id] += int(count)
            commitment_trace.append({
                "position": target.position,
                "action_id": decision.action.id,
                "spent_after_commit": budget - remaining_budget,
                "remaining_budget": remaining_budget,
                "expert_steps_after_commit": dict(used_steps),
            })
        return RoutePlan(
            decisions=tuple(decisions),
            spent=budget - remaining_budget,
            expert_calls=used_steps,
            algorithm="causal-online-relative-preference",
            diagnostics={
                "allocation_mode": "causal-online-committed",
                "global_optimal": False,
                "budget_unit": "Action.cost",
                "capacity_unit": "expert steps (sum of Action.steps)",
                "candidate_limit": 10,
                "probability_floor": self.probability_floor,
                "commitments": commitment_trace,
                "score_semantics": (
                    "relative MAP route preference; not measured task utility or expected NLL gain"
                ),
            },
        )

    def route(
        self,
        context: RoutingContext,
        actions: ActionOffers,
        *,
        budget: int | None = None,
        experts: Iterable[Expert] | None = None,
    ) -> RoutePlan:
        offered = _normalise_actions(context, actions)
        resolved_budget = self._budget(budget)
        selected_experts = tuple(experts) if experts is not None else context.experts
        if context.mode == "causal":
            return self._route_causal(
                context,
                offered,
                budget=resolved_budget,
                experts=selected_experts,
            )
        queries: list[tuple[int, str, dict]] = []
        for target in context.targets:
            action_list = offered[target.position]
            queries.append((
                target.position,
                _state_with_resources(
                    context,
                    target,
                    remaining_budget=resolved_budget,
                    experts=selected_experts,
                ),
                {
                    "id": f"joint:{target.position}",
                    "type": "choice",
                    "instructions": (
                        "Select exactly one next routing action for this target. "
                        "The listed costs are hard resource declarations; choose based on "
                        "the legal state and the action descriptions."
                    ),
                    "criteria": {action.id: _action_text(action) for action in action_list},
                },
            ))
        answers = _decide_grouped(self.backend, queries)
        options: list[AllocationOption] = []
        answer_by_action: dict[tuple[int, str], Mapping[str, float]] = {}
        for target in context.targets:
            identifier = f"joint:{target.position}"
            action_list = offered[target.position]
            probabilities = _probabilities(answers[identifier], (action.id for action in action_list))
            skip = next(action for action in action_list if action.expert_id is None)
            log_skip = math.log(max(probabilities[skip.id], self.probability_floor))
            for action in action_list:
                answer_by_action[(target.position, action.id)] = probabilities
                if action.expert_id is None:
                    options.append(AllocationOption(target.position, action, 0.0))
                if action.expert_id is not None:
                    preference = math.log(max(probabilities[action.id], self.probability_floor)) - log_skip
                    options.append(AllocationOption(target.position, action, preference))
            # The allocator inserts the zero-reference skip even when all
            # compute actions have negative relative preference.
        plan = allocate(
            options,
            selected_experts,
            budget=resolved_budget,
            max_states=self.max_states,
            algorithm="joint-map-relative-preference",
            diagnostics={
                "score_semantics": (
                    "log(max(p(action), eps)) - log(max(p(skip), eps)); "
                    "a relative MAP routing proxy, not task utility"
                ),
                "probability_floor": self.probability_floor,
                "candidate_limit": 10,
            },
        )
        return _attach_probabilities(
            plan,
            answer_by_action,
            algorithm="joint-map-relative-preference",
            diagnostics={},
        )


@dataclass(frozen=True)
class SignedGainBin:
    """One ordered Score level, with a heuristic gain value in caller units."""

    description: str
    gain: float

    def __post_init__(self):
        if not self.description.strip() or not math.isfinite(float(self.gain)):
            raise ValueError("Signed gain bins need a description and finite gain")


DEFAULT_SIGNED_GAIN_BINS: tuple[SignedGainBin, ...] = (
    SignedGainBin("large expected loss from taking this action", -3.0),
    SignedGainBin("moderate expected loss from taking this action", -1.5),
    SignedGainBin("small expected loss from taking this action", -0.5),
    SignedGainBin("approximately no expected change", 0.0),
    SignedGainBin("small expected gain from taking this action", 0.5),
    SignedGainBin("moderate expected gain from taking this action", 1.5),
    SignedGainBin("large expected gain from taking this action", 3.0),
)


class UtilityRouter(JointPlanRouter):
    """Elicit a signed gain distribution for each possible compute action.

    The score is the expectation over caller-defined signed gain bins and the
    declared cost remains a hard allocation budget.  The expectation is an
    elicited heuristic, not a measured task NLL or an empirically calibrated
    value estimate.
    """

    def __init__(
        self,
        backend: DecisionBackend,
        *,
        gain_bins: Sequence[SignedGainBin] = DEFAULT_SIGNED_GAIN_BINS,
        budget: int | None = None,
        max_states: int = 100_000,
    ):
        super().__init__(backend, budget=budget, max_states=max_states)
        bins = tuple(gain_bins)
        if not 1 <= len(bins) <= 10 or not all(isinstance(item, SignedGainBin) for item in bins):
            raise ValueError("Canopy Score requires 1..10 SignedGainBin values")
        if any(bins[i].gain >= bins[i + 1].gain for i in range(len(bins) - 1)):
            raise ValueError("Signed gain bins must be strictly ordered by gain")
        self.gain_bins = bins

    def _route_causal(
        self,
        context: RoutingContext,
        offered: Mapping[int, tuple[Action, ...]],
        *,
        budget: int,
        experts: Iterable[Expert],
    ) -> RoutePlan:
        base_experts = tuple(experts)
        remaining_budget = budget
        used_steps = {expert.id: 0 for expert in base_experts}
        decisions: list[RouteDecision] = []
        commitments: list[dict[str, Any]] = []
        level_keys = [str(index) for index in range(len(self.gain_bins))]
        for target in sorted(context.targets, key=lambda item: item.position):
            action_list = offered[target.position]
            state = _state_with_resources(
                context,
                target,
                remaining_budget=remaining_budget,
                experts=_residual_experts_from(base_experts, used_steps),
            )
            queries: list[tuple[int, str, dict]] = []
            by_id: dict[str, Action] = {}
            for action in action_list:
                if action.expert_id is None:
                    continue
                identifier = f"gain:{target.position}:{action.id}"
                by_id[identifier] = action
                queries.append((target.position, state, {
                    "id": identifier,
                    "type": "score",
                    "instructions": (
                        "Estimate the signed downstream gain of performing this one "
                        "routing action relative to skipping it. Do not treat the "
                        "ordinal levels as probabilities of correctness. " + _action_text(action)
                    ),
                    "criteria": [entry.description for entry in self.gain_bins],
                }))
            answers = _decide_grouped(self.backend, queries) if queries else {}
            skip = next(action for action in action_list if action.expert_id is None)
            local_options = [AllocationOption(target.position, skip, 0.0)]
            answer_by_action: dict[str, Mapping[str, float]] = {}
            for identifier, action in by_id.items():
                probabilities = _probabilities(answers[identifier], level_keys)
                answer_by_action[action.id] = probabilities
                expected_gain = sum(
                    probabilities[str(index)] * entry.gain
                    for index, entry in enumerate(self.gain_bins)
                )
                local_options.append(AllocationOption(target.position, action, expected_gain))
            local_plan = allocate(
                local_options,
                _residual_experts_from(base_experts, used_steps),
                budget=remaining_budget,
                max_states=self.max_states,
                algorithm="causal-online-utility-signed-gain",
                diagnostics={
                    "score_semantics": "elicited signed-gain heuristic, not observed NLL",
                },
            )
            decision = local_plan.decisions[0]
            decisions.append(replace(
                decision,
                probabilities=dict(answer_by_action.get(decision.action.id, {})),
            ))
            remaining_budget -= local_plan.spent
            for expert_id, count in local_plan.expert_calls.items():
                used_steps[expert_id] += int(count)
            commitments.append({
                "position": target.position,
                "action_id": decision.action.id,
                "spent_after_commit": budget - remaining_budget,
                "remaining_budget": remaining_budget,
                "expert_steps_after_commit": dict(used_steps),
            })
        return RoutePlan(
            decisions=tuple(decisions),
            spent=budget - remaining_budget,
            expert_calls=used_steps,
            algorithm="causal-online-utility-signed-gain",
            diagnostics={
                "allocation_mode": "causal-online-committed",
                "global_optimal": False,
                "budget_unit": "Action.cost",
                "capacity_unit": "expert steps (sum of Action.steps)",
                "score_semantics": (
                    "expectation over elicited signed-gain Score bins; not measured task utility"
                ),
                "gain_bins": [
                    {"description": item.description, "gain": item.gain} for item in self.gain_bins
                ],
                "commitments": commitments,
            },
        )

    def route(
        self,
        context: RoutingContext,
        actions: ActionOffers,
        *,
        budget: int | None = None,
        experts: Iterable[Expert] | None = None,
    ) -> RoutePlan:
        offered = _normalise_actions(context, actions, enforce_choice_limit=False)
        resolved_budget = self._budget(budget)
        selected_experts = tuple(experts) if experts is not None else context.experts
        if context.mode == "causal":
            return self._route_causal(
                context,
                offered,
                budget=resolved_budget,
                experts=selected_experts,
            )
        queries: list[tuple[int, str, dict]] = []
        compute_by_question: dict[str, tuple[int, Action]] = {}
        for target in context.targets:
            state = _state_with_resources(
                context,
                target,
                remaining_budget=resolved_budget,
                experts=selected_experts,
            )
            for action in offered[target.position]:
                if action.expert_id is None:
                    continue
                identifier = f"gain:{target.position}:{action.id}"
                compute_by_question[identifier] = (target.position, action)
                queries.append((
                    target.position,
                    state,
                    {
                        "id": identifier,
                        "type": "score",
                        "instructions": (
                            "Estimate the signed downstream gain of performing this one "
                            "routing action relative to skipping it. Do not treat the "
                            "ordinal levels as probabilities of correctness. "
                            + _action_text(action)
                        ),
                        "criteria": [entry.description for entry in self.gain_bins],
                    },
                ))
        answers = _decide_grouped(self.backend, queries) if queries else {}
        options: list[AllocationOption] = []
        answer_by_action: dict[tuple[int, str], Mapping[str, float]] = {}
        for identifier, (position, action) in compute_by_question.items():
            expected_keys = [str(index) for index in range(len(self.gain_bins))]
            probabilities = _probabilities(answers[identifier], expected_keys)
            expected_gain = sum(
                probabilities[str(index)] * item.gain
                for index, item in enumerate(self.gain_bins)
            )
            options.append(AllocationOption(position, action, expected_gain))
            answer_by_action[(position, action.id)] = probabilities
        for target in context.targets:
            skip = next(action for action in offered[target.position] if action.expert_id is None)
            options.append(AllocationOption(target.position, skip, 0.0))
        plan = allocate(
            options,
            selected_experts,
            budget=resolved_budget,
            max_states=self.max_states,
            algorithm="utility-signed-gain",
            diagnostics={
                "score_semantics": (
                    "expectation over elicited signed-gain Score bins; a heuristic "
                    "gain proxy, not observed NLL or calibrated task utility"
                ),
                "gain_bins": [
                    {"description": item.description, "gain": item.gain} for item in self.gain_bins
                ],
                "budget_role": "hard constraint in Action.cost units",
            },
        )
        return _attach_probabilities(
            plan,
            answer_by_action,
            algorithm="utility-signed-gain",
            diagnostics={},
        )


def _residual_experts(context: RoutingContext, used_steps: Mapping[str, int]) -> tuple[Expert, ...]:
    return _residual_experts_from(context.experts, used_steps)


def _update_context_after_pass(
    context: RoutingContext,
    plan: RoutePlan,
    pass_index: int,
    observed: Mapping[int, Mapping[str, Any]] | None,
) -> RoutingContext:
    by_position = {decision.position: decision for decision in plan.decisions}
    targets: list[TokenState] = []
    for target in context.targets:
        decision = by_position.get(target.position)
        if decision is None or decision.action.expert_id is None:
            targets.append(target)
            continue
        observation = dict(target.observation)
        observation["last_routing_action"] = decision.action.id
        observation["completed_routing_steps"] = target.depth + decision.action.steps
        if observed and target.position in observed:
            supplied = observed[target.position]
            if not isinstance(supplied, Mapping):
                raise ValueError("Execution observations must be mappings by target position")
            observation["execution_observation"] = dict(supplied)
        targets.append(replace(target, depth=target.depth + decision.action.steps, observation=observation))
    history = list(context.history)
    for decision in plan.decisions:
        if decision.action.expert_id is None:
            continue
        history.append({
            "visible_after": decision.position,
            "kind": "routing_execution",
            "pass": pass_index,
            "position": decision.position,
            "action_id": decision.action.id,
            "expert_id": decision.action.expert_id,
            "steps": decision.action.steps,
            "cost": decision.action.cost,
            "receipt": dict(observed.get(decision.position, {})) if observed else {},
        })
    return replace(context, targets=tuple(targets), history=tuple(history))


def _prefix_view(
    context: RoutingContext, position: int, experts: Iterable[Expert]
) -> RoutingContext:
    """Build the only context object exposed to causal host callbacks."""
    if context.mode != "causal":
        raise ValueError("A prefix view is only defined for causal routing")
    return replace(
        context,
        tokens=context.tokens[:position + 1],
        targets=tuple(target for target in context.targets if target.position <= position),
        experts=tuple(experts),
        history=tuple(entry for entry in context.history if entry["visible_after"] <= position),
        metadata=tuple(entry for entry in context.metadata if entry["visible_after"] <= position),
    )


def _offers_for_positions(offers: ActionOffers, positions: Iterable[int]) -> ActionOffers:
    """Filter a static per-target offer map before giving it to a restricted view."""
    if isinstance(offers, Mapping):
        wanted = set(positions)
        return {position: value for position, value in offers.items() if position in wanted}
    return offers


def _append_stop_history(context: RoutingContext, decisions: Iterable[RouteDecision], pass_index: int) -> RoutingContext:
    records = list(context.history)
    for decision in decisions:
        if decision.action.expert_id is None:
            records.append({
                "visible_after": decision.position,
                "kind": "routing_stop",
                "pass": pass_index,
                "position": decision.position,
                "action_id": decision.action.id,
                "score": decision.score,
            })
    return replace(context, history=tuple(records))


class AdaptiveRecursionRouter(JointPlanRouter):
    """Re-query after each executed pass while enforcing cumulative resources.

    ``executor`` must receive ``(current_context, pass_plan)`` or expose an
    ``execute(current_context, pass_plan)`` method.  A canonical callback
    executor is supplied in :mod:`jev_routing.execution`; it receives only each
    target's ``for_target`` state.  If an executor returns a mapping, or an
    object with an ``observations`` mapping, those observations become visible
    only at that target position on the next pass.
    """

    def __init__(
        self,
        backend: DecisionBackend,
        *,
        max_passes: int,
        budget: int | None = None,
        max_states: int = 100_000,
        probability_floor: float = 1e-12,
    ):
        super().__init__(
            backend,
            budget=budget,
            max_states=max_states,
            probability_floor=probability_floor,
        )
        _integer(max_passes, "max_passes", 1)
        self.max_passes = max_passes

    @staticmethod
    def _run_executor(executor: Any, context: RoutingContext, plan: RoutePlan) -> Mapping[int, Mapping[str, Any]] | None:
        if executor is None:
            raise ValueError(
                "Adaptive recursion selected compute work but no executor was supplied; "
                "use CausalCallbackExecutor or an executor with execute(context, plan)"
            )
        if not hasattr(executor, "execute"):
            raise TypeError(
                "Raw adaptive callbacks are rejected because they would receive a full context; "
                "wrap the callback in CausalCallbackExecutor"
            )
        outcome = executor.execute(context, plan)
        if outcome is None:
            return None
        if hasattr(outcome, "observations"):
            outcome = outcome.observations
        if not isinstance(outcome, Mapping):
            raise ValueError("Adaptive executor must return an observations mapping or None")
        return outcome

    def _causal_route(
        self,
        context: RoutingContext,
        actions: PerPassOffers,
        *,
        executor: Any,
        budget: int,
    ) -> RoutePlan:
        """Finish all recursive work for each target before exposing a suffix.

        This ordering makes both route preferences and remaining resources for
        an early target independent of later target scores.  Later targets may
        observe earlier committed receipts through their legal prefix.
        """
        current = context
        remaining_budget = budget
        used_steps = {expert.id: 0 for expert in context.experts}
        decisions: list[RouteDecision] = []
        receipts: list[dict[str, Any]] = []
        targets_in_order = tuple(sorted(context.targets, key=lambda target: target.position))
        for initial_target in targets_in_order:
            completed = False
            for local_pass in range(self.max_passes):
                current_target = next(
                    target for target in current.targets if target.position == initial_target.position
                )
                prefix = _prefix_view(
                    current,
                    current_target.position,
                    _residual_experts_from(context.experts, used_steps),
                )
                pass_offers = actions(prefix, local_pass) if callable(actions) else actions
                offered = _normalise_actions(
                    prefix,
                    _offers_for_positions(pass_offers, (current_target.position,)),
                )
                # A zero remaining cost budget is absorbing because all compute
                # Actions require positive cost by schema contract.
                if remaining_budget == 0:
                    skip = next(action for action in offered[current_target.position] if action.expert_id is None)
                    decision = RouteDecision(current_target.position, skip, 0.0, {})
                    decisions.append(decision)
                    receipts.append({
                        "position": current_target.position,
                        "pass": local_pass,
                        "selected_compute_actions": 0,
                        "absorbing_reason": "budget_exhausted",
                    })
                    current = _append_stop_history(current, (decision,), local_pass)
                    completed = True
                    break
                pass_plan = self._route_causal(
                    prefix,
                    offered,
                    budget=remaining_budget,
                    experts=prefix.experts,
                    target_positions=(current_target.position,),
                )
                decision = pass_plan.decisions[0]
                decisions.append(decision)
                receipt = {
                    "position": current_target.position,
                    "pass": local_pass,
                    "action_id": decision.action.id,
                    "spent": pass_plan.spent,
                    "expert_steps": dict(pass_plan.expert_calls),
                    "selected_compute_actions": int(decision.action.expert_id is not None),
                }
                receipts.append(receipt)
                if decision.action.expert_id is None:
                    current = _append_stop_history(current, (decision,), local_pass)
                    completed = True  # skip is an absorbing stop decision
                    break
                observed = self._run_executor(executor, prefix, pass_plan)
                current = _update_context_after_pass(current, pass_plan, local_pass, observed)
                remaining_budget -= pass_plan.spent
                for expert_id, count in pass_plan.expert_calls.items():
                    used_steps[expert_id] += int(count)
            if not completed and remaining_budget == 0:
                # The next target is handled by the explicit absorbing branch.
                continue
        return RoutePlan(
            decisions=tuple(decisions),
            spent=budget - remaining_budget,
            expert_calls=used_steps,
            algorithm="causal-adaptive-recursion-relative-preference",
            diagnostics={
                "allocation_mode": "causal-online-per-target-recursion",
                "global_optimal": False,
                "passes": receipts,
                "max_passes": self.max_passes,
                "budget_unit": "Action.cost",
                "capacity_unit": "expert steps (sum of Action.steps)",
                "requery_contract": (
                    "A target is re-queried only after its actual execution receipt; "
                    "skip is absorbing and later targets never affect earlier commitments."
                ),
                "score_semantics": "Relative MAP route preference, not task utility.",
            },
        )

    def route(
        self,
        context: RoutingContext,
        actions: PerPassOffers,
        *,
        executor: Any = None,
        budget: int | None = None,
    ) -> RoutePlan:
        remaining_budget = self._budget(budget)
        if context.mode == "causal":
            return self._causal_route(
                context, actions, executor=executor, budget=remaining_budget
            )
        current = context
        used_steps = {expert.id: 0 for expert in context.experts}
        active_positions = {target.position for target in context.targets}
        all_decisions: list[RouteDecision] = []
        pass_receipts: list[dict[str, Any]] = []
        total_spent = 0
        for pass_index in range(self.max_passes):
            if not active_positions:
                break
            active_context = replace(
                current,
                targets=tuple(
                    target for target in current.targets if target.position in active_positions
                ),
            )
            pass_offers = actions(active_context, pass_index) if callable(actions) else actions
            # Call the parent method directly: its single pass budget and expert
            # capacities are the *remaining* hard resources, not a new budget.
            pass_plan = JointPlanRouter.route(
                self,
                active_context,
                _offers_for_positions(pass_offers, active_positions),
                budget=remaining_budget,
                experts=_residual_experts(context, used_steps),
            )
            compute = [decision for decision in pass_plan.decisions if decision.action.expert_id is not None]
            stopped = [decision for decision in pass_plan.decisions if decision.action.expert_id is None]
            all_decisions.extend(pass_plan.decisions)
            pass_receipts.append({
                "pass": pass_index,
                "spent": pass_plan.spent,
                "expert_steps": dict(pass_plan.expert_calls),
                "selected_compute_actions": len(compute),
            })
            if not compute:
                current = _append_stop_history(current, stopped, pass_index)
                active_positions.difference_update(decision.position for decision in stopped)
                break
            current = _append_stop_history(current, stopped, pass_index)
            active_positions.difference_update(decision.position for decision in stopped)
            observed = self._run_executor(executor, active_context, pass_plan)
            current = _update_context_after_pass(current, pass_plan, pass_index, observed)
            remaining_budget -= pass_plan.spent
            total_spent += pass_plan.spent
            for expert_id, count in pass_plan.expert_calls.items():
                used_steps[expert_id] += int(count)
            if remaining_budget == 0:
                break
        return RoutePlan(
            decisions=tuple(all_decisions),
            spent=total_spent,
            expert_calls=used_steps,
            algorithm="adaptive-recursion-relative-preference",
            diagnostics={
                "passes": pass_receipts,
                "max_passes": self.max_passes,
                "budget_unit": "Action.cost",
                "capacity_unit": "expert steps (sum of Action.steps)",
                "requery_contract": "Each pass uses the current legal target state and execution receipt.",
                "score_semantics": (
                    "Each pass uses relative MAP route preference; it is not task utility."
                ),
            },
        )
