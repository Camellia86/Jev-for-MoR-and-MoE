"""Exact, resource-constrained allocation of token-level routing actions.

The allocator is deliberately unaware of how a score was produced.  A caller
may pass an elicited utility, a calibrated reward model, or a *labelled proxy*
such as a relative route preference.  Costs are integer units declared by an
``Action`` and expert capacities are integer **expert-step** units.  This keeps
probabilities from silently becoming task utility inside the allocator.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

from .schema import Action, AllocationOption, Expert, RouteDecision, RoutePlan


class AllocationError(ValueError):
    """The offered actions cannot be allocated under the stated contract."""


class StateLimitExceeded(AllocationError):
    """Exact DP would exceed the caller's explicitly bounded state budget."""


@dataclass(frozen=True)
class _PartialPlan:
    score: float
    chosen: tuple[AllocationOption, ...]


def _validate_integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise AllocationError(f"{name} must be an integer >= {minimum}")


def _as_groups(options: Iterable[AllocationOption]) -> dict[int, list[AllocationOption]]:
    grouped: dict[int, list[AllocationOption]] = defaultdict(list)
    for option in options:
        if not isinstance(option, AllocationOption):
            raise AllocationError("options must contain AllocationOption values")
        if type(option.position) is not int or option.position < 0:
            raise AllocationError("Allocation positions must be nonnegative integers")
        if not isinstance(option.action, Action):
            raise AllocationError("Allocation options must contain Action values")
        if not isinstance(option.value, (float, int)) or not math.isfinite(option.value):
            raise AllocationError("Allocation option values must be finite numbers")
        grouped[option.position].append(option)
    if not grouped:
        raise AllocationError("At least one target position is required")
    return dict(grouped)


def _normalise_groups(
    options: Iterable[AllocationOption], experts: Iterable[Expert]
) -> tuple[dict[int, tuple[AllocationOption, ...]], tuple[Expert, ...]]:
    expert_list = tuple(experts)
    if not expert_list or any(not isinstance(expert, Expert) for expert in expert_list):
        raise AllocationError("experts must be a nonempty iterable of Expert values")
    expert_ids = [expert.id for expert in expert_list]
    if len(set(expert_ids)) != len(expert_ids):
        raise AllocationError("Expert IDs must be unique")
    known = set(expert_ids)
    result: dict[int, tuple[AllocationOption, ...]] = {}
    for position, offered in _as_groups(options).items():
        seen: set[str] = set()
        cleaned: list[AllocationOption] = []
        has_skip = False
        for option in offered:
            action = option.action
            if action.id in seen:
                raise AllocationError(
                    f"Position {position} offers duplicate action ID {action.id!r}"
                )
            seen.add(action.id)
            if action.expert_id is not None and action.expert_id not in known:
                raise AllocationError(
                    f"Action {action.id!r} names unknown expert {action.expert_id!r}"
                )
            if action.expert_id is None:
                has_skip = True
                # No work is the reference allocation.  A negative or positive
                # score here would let a caller redefine the no-op baseline.
                if float(option.value) != 0.0:
                    raise AllocationError("A skip action must have allocation value 0")
            else:
                cleaned.append(
                    AllocationOption(position, action, float(option.value))
                )
        if not has_skip:
            cleaned.append(AllocationOption(position, Action.skip(), 0.0))
        else:
            # Preserve an explicitly named skip so route output can retain its
            # caller-visible action ID, after checking it is a true no-op.
            cleaned.extend(
                AllocationOption(position, option.action, 0.0)
                for option in offered
                if option.action.expert_id is None
            )
        result[position] = tuple(cleaned)
    return result, expert_list


def _is_better(candidate: _PartialPlan, incumbent: _PartialPlan | None) -> bool:
    if incumbent is None:
        return True
    if candidate.score != incumbent.score:
        return candidate.score > incumbent.score
    # Exact resource keys already fix total cost and capacity.  A stable lexical
    # tie break makes replay fixtures and audits deterministic.
    candidate_ids = tuple(option.action.id for option in candidate.chosen)
    incumbent_ids = tuple(option.action.id for option in incumbent.chosen)
    return candidate_ids < incumbent_ids


def allocate(
    options: Iterable[AllocationOption],
    experts: Iterable[Expert],
    *,
    budget: int,
    max_states: int = 100_000,
    algorithm: str = "exact-resource-dp",
    diagnostics: Mapping[str, object] | None = None,
) -> RoutePlan:
    """Choose exactly one action per target by exact multiple-choice DP.

    ``budget`` is in ``Action.cost`` units.  Each finite ``Expert.capacity`` is
    consumed by ``Action.steps`` (not by the number of selected rows).  The DP
    keeps every reachable integer resource vector, so it is exact; it fails
    visibly instead of applying an unannounced approximate pruning rule when
    ``max_states`` would be exceeded.
    """
    _validate_integer(budget, "budget")
    _validate_integer(max_states, "max_states", 1)
    groups, expert_list = _normalise_groups(options, experts)
    constrained = tuple(expert for expert in expert_list if expert.capacity is not None)
    constrained_ids = tuple(expert.id for expert in constrained)
    capacity = tuple(int(expert.capacity) for expert in constrained)
    index_by_expert = {identifier: index for index, identifier in enumerate(constrained_ids)}

    # A key is (spent cost, used steps for each finite-capacity expert).  The
    # value stores the best score among paths with exactly those resources.
    states: dict[tuple[int, tuple[int, ...]], _PartialPlan] = {
        (0, tuple(0 for _ in constrained)): _PartialPlan(0.0, ())
    }
    for position in sorted(groups):
        next_states: dict[tuple[int, tuple[int, ...]], _PartialPlan] = {}
        for (spent, used), partial in states.items():
            for option in groups[position]:
                action = option.action
                next_spent = spent + action.cost
                if next_spent > budget:
                    continue
                next_used = list(used)
                if action.expert_id in index_by_expert:
                    expert_index = index_by_expert[action.expert_id]
                    next_used[expert_index] += action.steps
                    if next_used[expert_index] > capacity[expert_index]:
                        continue
                key = (next_spent, tuple(next_used))
                candidate = _PartialPlan(
                    partial.score + float(option.value), partial.chosen + (option,)
                )
                incumbent = next_states.get(key)
                if incumbent is None and len(next_states) >= max_states:
                    raise StateLimitExceeded(
                        f"Exact resource DP would exceed max_states={max_states} after "
                        f"position {position}; no approximate pruning was applied"
                    )
                if _is_better(candidate, incumbent):
                    next_states[key] = candidate
        if not next_states:
            raise AllocationError(
                f"No legal allocation remains after target position {position}"
            )
        states = next_states

    best_key: tuple[int, tuple[int, ...]] | None = None
    best: _PartialPlan | None = None
    for key, partial in states.items():
        if best is None or best_key is None or partial.score > best.score:
            best_key, best = key, partial
            continue
        if partial.score != best.score:
            continue
        # Across different resource vectors, do not allow a lexical action ID
        # to pay unnecessary cost.  The exact final order is objective, then
        # lower cost, then lexical replay tie-break.
        if key[0] < best_key[0]:
            best_key, best = key, partial
        elif key[0] == best_key[0]:
            candidate_ids = tuple(option.action.id for option in partial.chosen)
            incumbent_ids = tuple(option.action.id for option in best.chosen)
            if candidate_ids < incumbent_ids:
                best_key, best = key, partial
    if best is None or best_key is None:  # defensive; every group has a skip.
        raise AllocationError("Exact resource DP did not produce a plan")

    usage = {expert.id: 0 for expert in expert_list}
    for option in best.chosen:
        if option.action.expert_id is not None:
            usage[option.action.expert_id] += option.action.steps
    combined_diagnostics = {
        "objective": "sum of caller-supplied AllocationOption.value",
        "budget_unit": "Action.cost",
        "capacity_unit": "expert steps (sum of Action.steps)",
        "exact": True,
        "state_count": len(states),
        "max_states": max_states,
    }
    if diagnostics:
        combined_diagnostics.update(dict(diagnostics))
    return RoutePlan(
        decisions=tuple(
            RouteDecision(option.position, option.action, float(option.value))
            for option in best.chosen
        ),
        spent=best_key[0],
        # Schema retains this historical field name.  Its values are step units,
        # which is made explicit above and in the README-level API docs.
        expert_calls=usage,
        algorithm=algorithm,
        diagnostics=combined_diagnostics,
    )


exact_allocate = allocate
