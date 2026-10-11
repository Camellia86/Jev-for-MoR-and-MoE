"""Sparse routing executors with auditable work ledgers.

The tensor executor is intentionally small: a selected expert returns a
residual update for selected token rows, while all unselected rows are copied
unchanged.  The callback executor is the canonical bridge to a host model that
needs an explicit causal information boundary for every token-level action.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Optional, Sequence, Tuple, Union

from .schema import RouteDecision, RoutePlan, RoutingContext


@dataclass(frozen=True)
class ExecutionRecord:
    position: int
    action_id: str
    expert_id: str
    depth: int
    step: int


@dataclass(frozen=True)
class ExecutionResult:
    hidden: Any
    records: Tuple[ExecutionRecord, ...]
    observations: Mapping[int, Mapping[str, Any]]


def _validate_plan(
    plan: RoutePlan,
    *,
    capacities: Optional[Mapping[str, Optional[int]]] = None,
    budget: Optional[int] = None,
) -> None:
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("budget must be a nonnegative integer")
    spent = sum(decision.action.cost for decision in plan.decisions)
    if spent != plan.spent:
        raise ValueError("RoutePlan.spent must equal the selected Action.cost sum")
    if budget is not None and spent > budget:
        raise ValueError("RoutePlan exceeds the supplied Action.cost budget")
    used: dict[str, int] = {}
    for decision in plan.decisions:
        if type(decision.position) is not int or decision.position < 0:
            raise ValueError("RoutePlan positions must be nonnegative integers")
        action = decision.action
        if action.expert_id is not None:
            used[action.expert_id] = used.get(action.expert_id, 0) + action.steps
    reported = dict(plan.expert_calls)
    if any(type(value) is not int or value < 0 for value in reported.values()):
        raise ValueError("RoutePlan expert step counts must be nonnegative integers")
    for expert_id, count in used.items():
        if reported.get(expert_id) != count:
            raise ValueError("RoutePlan expert_calls must count Action.steps exactly")
    if any(expert_id not in used and count != 0 for expert_id, count in reported.items()):
        raise ValueError("RoutePlan reports nonzero steps for an unselected expert")
    if capacities is not None:
        for expert_id, count in used.items():
            if expert_id not in capacities:
                raise ValueError(f"RoutePlan names executor-unknown expert {expert_id!r}")
            capacity = capacities[expert_id]
            if capacity is not None and count > capacity:
                raise ValueError(f"RoutePlan exceeds expert-step capacity for {expert_id!r}")


class SparseMoEExecutor:  # torch base class is assigned below only when imported
    """Placeholder used to keep importing this module free of a torch import."""

    def __init__(self, *args, **kwargs):
        raise ImportError("SparseMoEExecutor requires PyTorch; install the optional training dependency")


def _build_sparse_executor():
    import torch
    from torch import nn

    class _SparseMoEExecutor(nn.Module):
        """Residual sparse expert primitive, not a complete MoR/attention block."""

        def __init__(
            self,
            experts: Union[Mapping[str, nn.Module], nn.ModuleDict],
            *,
            capacities: Optional[Mapping[str, Optional[int]]] = None,
        ):
            super().__init__()
            if isinstance(experts, nn.ModuleDict):
                modules = dict(experts)
            else:
                modules = dict(experts)
            if not modules or not all(isinstance(key, str) and key for key in modules):
                raise ValueError("Provide a nonempty mapping of named torch experts")
            if not all(isinstance(module, nn.Module) for module in modules.values()):
                raise TypeError("Every sparse executor expert must be an nn.Module")
            self.experts = nn.ModuleDict(modules)
            supplied = {} if capacities is None else dict(capacities)
            unknown = set(supplied) - set(modules)
            if unknown:
                raise ValueError(f"Capacities name unknown experts: {sorted(unknown)}")
            for capacity in supplied.values():
                if capacity is not None and (type(capacity) is not int or capacity < 0):
                    raise ValueError("Expert capacities must be nonnegative integers or None")
            self.capacities = {name: supplied.get(name) for name in modules}

        def execute(
            self,
            hidden: torch.Tensor,
            plan: RoutePlan,
            *,
            budget: Optional[int] = None,
            depths: Optional[Mapping[int, int]] = None,
        ) -> ExecutionResult:
            if not isinstance(hidden, torch.Tensor) or hidden.ndim != 2:
                raise ValueError("SparseMoEExecutor expects hidden with shape [tokens, hidden_size]")
            _validate_plan(plan, capacities=self.capacities, budget=budget)
            selected = [decision for decision in plan.decisions if decision.action.expert_id is not None]
            positions = [decision.position for decision in selected]
            if len(set(positions)) != len(positions):
                raise ValueError(
                    "Tensor execution accepts at most one compute action per token; "
                    "use one adaptive pass at a time for recursive actions"
                )
            if any(decision.position >= hidden.shape[0] for decision in plan.decisions):
                raise ValueError("A route position is outside hidden's token axis")
            output = hidden.clone()
            records = []
            by_expert: dict[str, list[RouteDecision]] = {}
            for decision in selected:
                by_expert.setdefault(decision.action.expert_id, []).append(decision)
            # Each expert's active rows are batched on every recursive step.
            for expert_id, decisions in by_expert.items():
                expert = self.experts[expert_id]
                maximum = max(decision.action.steps for decision in decisions)
                for step in range(maximum):
                    active = [decision for decision in decisions if step < decision.action.steps]
                    indices = torch.tensor(
                        [decision.position for decision in active], device=hidden.device, dtype=torch.long
                    )
                    current = output.index_select(0, indices)
                    delta = expert(current)
                    if not isinstance(delta, torch.Tensor) or delta.shape != current.shape:
                        raise ValueError(
                            f"Expert {expert_id!r} must return a residual tensor with shape {tuple(current.shape)}"
                        )
                    output.index_copy_(0, indices, current + delta)
                    for decision in active:
                        base_depth = 0 if depths is None else int(depths.get(decision.position, 0))
                        records.append(ExecutionRecord(
                            position=decision.position,
                            action_id=decision.action.id,
                            expert_id=expert_id,
                            depth=base_depth + step + 1,
                            step=step + 1,
                        ))
            return ExecutionResult(output, tuple(records), {})

        def forward(self, hidden: torch.Tensor, plan: RoutePlan) -> torch.Tensor:
            return self.execute(hidden, plan).hidden

    return _SparseMoEExecutor


# Importing the package does not require CUDA; torch is required only for this
# executor.  The project itself declares torch for users who instantiate it.
try:  # pragma: no cover - the fallback is only for documentation-only installs.
    SparseMoEExecutor = _build_sparse_executor()
except ModuleNotFoundError:  # pragma: no cover
    pass


class CausalCallbackExecutor:
    """Execute callbacks with only the serialized legal state for each target.

    The callback is called as ``callback(state=..., position=..., action=...,
    depth=...)``.  It never receives ``RoutingContext`` or raw token arrays, so
    an ordinary callback cannot inspect a suffix by accident.  It may return a
    JSON-like mapping, which becomes that target's next-pass observation.
    """

    def __init__(self, callback: Callable[..., Optional[Mapping[str, Any]]]):
        if not callable(callback):
            raise TypeError("callback must be callable")
        self.callback = callback

    def execute(self, context: RoutingContext, plan: RoutePlan) -> ExecutionResult:
        capacities = {expert.id: expert.capacity for expert in context.experts}
        _validate_plan(plan, capacities=capacities)
        current_context = context
        observations: dict[int, Mapping[str, Any]] = {}
        records = []
        for decision in plan.decisions:
            action = decision.action
            if action.expert_id is None:
                continue
            targets = {target.position: target for target in current_context.targets}
            target = targets.get(decision.position)
            if target is None:
                raise ValueError("RoutePlan names a callback target outside the context")
            for step in range(action.steps):
                depth = target.depth + 1
                # Recreate the legal prefix for every recursive callback.  The
                # previous callback's receipt and completed depth have already
                # been committed into ``current_context`` below.
                legal_state = current_context.for_target(target)
                payload = self.callback(
                    state=legal_state,
                    position=target.position,
                    action=action,
                    depth=depth,
                )
                if payload is not None:
                    if not isinstance(payload, Mapping):
                        raise ValueError("Causal callback results must be mappings or None")
                    merged = dict(observations.get(target.position, {}))
                    merged.update(dict(payload))
                    observations[target.position] = merged
                updated_observation = dict(target.observation)
                updated_observation.update({
                    "last_routing_action": action.id,
                    "completed_routing_steps": depth,
                })
                if target.position in observations:
                    updated_observation["execution_observation"] = dict(observations[target.position])
                updated_target = type(target)(
                    position=target.position,
                    depth=depth,
                    observation=updated_observation,
                )
                current_context = type(current_context)(
                    tokens=current_context.tokens,
                    targets=tuple(
                        updated_target if item.position == target.position else item
                        for item in current_context.targets
                    ),
                    experts=current_context.experts,
                    task=current_context.task,
                    mode=current_context.mode,
                    history=(*current_context.history, {
                        "visible_after": target.position,
                        "kind": "callback_execution_step",
                        "position": target.position,
                        "action_id": action.id,
                        "expert_id": action.expert_id,
                        "depth": depth,
                        "receipt": dict(observations.get(target.position, {})),
                    }),
                    metadata=current_context.metadata,
                )
                target = updated_target
                records.append(ExecutionRecord(
                    position=target.position,
                    action_id=action.id,
                    expert_id=action.expert_id,
                    depth=depth,
                    step=step + 1,
                ))
        return ExecutionResult(None, tuple(records), observations)
