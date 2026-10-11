"""Explicit computation contracts; costs and capacities count declared work units."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping
import json


def _integer(value: int, name: str, minimum: int = 0) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _json(value: Any) -> None:
    json.dumps(value, allow_nan=False)


@dataclass(frozen=True)
class Expert:
    id: str
    description: str
    capacity: int | None = None

    def __post_init__(self):
        if not self.id or not self.description.strip():
            raise ValueError("Experts require an ID and a capability description")
        if self.capacity is not None:
            _integer(self.capacity, "capacity")


@dataclass(frozen=True)
class Action:
    id: str
    expert_id: str | None
    steps: int
    cost: int
    description: str = ""

    def __post_init__(self):
        if not self.id:
            raise ValueError("Action IDs must be nonempty")
        _integer(self.steps, "steps")
        _integer(self.cost, "cost")
        if self.expert_id is None:
            if self.steps != 0 or self.cost != 0:
                raise ValueError("Skip actions must have zero steps and cost")
        elif not self.expert_id or self.steps < 1 or self.cost < 1:
            raise ValueError("Compute actions need an expert, positive steps and cost")

    @classmethod
    def skip(cls) -> Action:
        return cls("skip", None, 0, 0, "Keep the current state; perform no additional expert work.")


@dataclass(frozen=True)
class TokenState:
    position: int
    depth: int = 0
    observation: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        _integer(self.position, "position")
        _integer(self.depth, "depth")
        if not isinstance(self.observation, Mapping):
            raise ValueError("Observations must be JSON mappings")
        _json(dict(self.observation))


@dataclass(frozen=True)
class RoutingContext:
    tokens: tuple[str, ...]
    targets: tuple[TokenState, ...]
    experts: tuple[Expert, ...]
    task: str
    mode: str = "causal"
    history: tuple[dict[str, Any], ...] = ()
    metadata: tuple[dict[str, Any], ...] = ()

    def __post_init__(self):
        if self.mode not in {"causal", "bidirectional"}:
            raise ValueError("mode must be causal or bidirectional")
        if not self.tokens or not all(isinstance(t, str) for t in self.tokens):
            raise ValueError("Provide every visible token as an explicit string")
        if not self.targets or len({t.position for t in self.targets}) != len(self.targets):
            raise ValueError("Targets must be nonempty and have unique positions")
        if any(t.position >= len(self.tokens) for t in self.targets):
            raise ValueError("Target position is outside the provided context")
        if not self.experts or len({e.id for e in self.experts}) != len(self.experts):
            raise ValueError("Expert IDs must be nonempty and unique")
        if not self.task.strip():
            raise ValueError("An ex-ante task instruction is required")
        for entry in (*self.history, *self.metadata):
            if not isinstance(entry, dict) or "visible_after" not in entry:
                raise ValueError("Every history/metadata record requires visible_after")
            _integer(entry["visible_after"], "visible_after")
            _json(entry)

    def for_target(self, target: TokenState | int) -> str:
        from .context import make_state
        return make_state(self, target)


@dataclass(frozen=True)
class AllocationOption:
    position: int
    action: Action
    value: float


@dataclass(frozen=True)
class RouteDecision:
    position: int
    action: Action
    score: float
    probabilities: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class RoutePlan:
    decisions: tuple[RouteDecision, ...]
    spent: int
    expert_calls: Mapping[str, int]
    algorithm: str = ""
    diagnostics: Mapping[str, Any] = field(default_factory=dict)
