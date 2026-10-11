"""Optional supervised distillation utilities for a trainable local router.

These helpers distil a supplied teacher distribution (for example, a recorded
Canopy route decision).  They do not claim that the teacher probabilities are
task value, and they never train or modify Canopy-Jev itself.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from typing import Optional


@dataclass(frozen=True)
class RoutingDistillationExample:
    """A provenance-bound student-router target recorded from one JEV query."""

    state: str
    candidate_ids: tuple[str, ...]
    teacher_probabilities: tuple[float, ...]
    context_digest: str
    target_position: int
    mode: str

    def __post_init__(self):
        if not isinstance(self.state, str) or not self.state:
            raise ValueError("A distillation example requires the serialized decision state")
        if not isinstance(self.context_digest, str) or len(self.context_digest) != 64:
            raise ValueError("context_digest must be a SHA-256 hex digest")
        if hashlib.sha256(self.state.encode("utf-8")).hexdigest() != self.context_digest:
            raise ValueError("context_digest does not match the recorded state")
        if type(self.target_position) is not int or self.target_position < 0:
            raise ValueError("target_position must be a nonnegative integer")
        if self.mode not in {"causal", "bidirectional"}:
            raise ValueError("mode must be causal or bidirectional")
        if not 1 <= len(self.candidate_ids) <= 10 or len(set(self.candidate_ids)) != len(self.candidate_ids):
            raise ValueError("candidate_ids must contain 1..10 unique IDs")
        if len(self.teacher_probabilities) != len(self.candidate_ids):
            raise ValueError("teacher probabilities must align with candidate_ids")
        if not all(isinstance(value, (float, int)) and math.isfinite(value) and value >= 0
                   for value in self.teacher_probabilities):
            raise ValueError("teacher probabilities must be finite and nonnegative")
        if not math.isclose(sum(self.teacher_probabilities), 1.0, abs_tol=2e-5):
            raise ValueError("teacher probabilities must sum to one")
        try:
            payload = json.loads(self.state)
        except json.JSONDecodeError as error:
            raise ValueError("state must be a serialized routing context") from error
        if payload.get("mode") != self.mode:
            raise ValueError("Recorded state mode does not match example mode")
        if payload.get("target", {}).get("position") != self.target_position:
            raise ValueError("Recorded state target does not match target_position")
        if self.mode == "causal" and payload.get("visible_through") != self.target_position:
            raise ValueError("Causal student examples must end exactly at target_position")


def routing_distillation_loss(
    student_logits,
    teacher_probabilities,
    *,
    candidate_mask=None,
    example_weights=None,
    temperature: float = 1.0,
):
    """Return masked KL(teacher || student) for aligned candidate axes.

    ``candidate_mask`` makes padded candidates impossible for both sides.  The
    function checks that all probability mass belongs to legal candidates so a
    training pipeline cannot silently teach a different action set than the
    recorded JEV decision.
    """
    import torch
    from torch.nn import functional as F

    if not isinstance(student_logits, torch.Tensor) or not isinstance(teacher_probabilities, torch.Tensor):
        raise TypeError("student_logits and teacher_probabilities must be torch tensors")
    if (student_logits.ndim != 2 or student_logits.shape[0] == 0 or student_logits.shape[1] == 0
            or teacher_probabilities.shape != student_logits.shape):
        raise ValueError("Expected matching [batch, candidates] logits and teacher probabilities")
    if (not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature))
            or not 0.0 < float(temperature)):
        raise ValueError("temperature must be positive")
    # The target is a recorded decision label.  Detaching prevents an accidental
    # training graph from flowing into an online teacher or a preprocessing step.
    teacher = teacher_probabilities.detach().to(
        device=student_logits.device, dtype=student_logits.dtype
    )
    if candidate_mask is None:
        mask = torch.ones_like(student_logits, dtype=torch.bool)
    else:
        if not isinstance(candidate_mask, torch.Tensor) or candidate_mask.shape != student_logits.shape:
            raise ValueError("candidate_mask must have shape [batch, candidates]")
        mask = candidate_mask.to(device=student_logits.device, dtype=torch.bool)
    if not torch.all(mask.any(dim=-1)):
        raise ValueError("Every training row needs at least one legal candidate")
    if not torch.isfinite(student_logits.masked_select(mask)).all():
        raise ValueError("student logits must be finite on legal candidates")
    if not torch.isfinite(teacher).all() or (teacher < 0).any():
        raise ValueError("teacher probabilities must be finite and nonnegative")
    if (teacher.masked_select(~mask) != 0).any():
        raise ValueError("teacher probability mass may not appear on masked candidates")
    normalizer = teacher.sum(dim=-1, keepdim=True)
    if not torch.allclose(normalizer, torch.ones_like(normalizer), atol=2e-5, rtol=0.0):
        raise ValueError("Each teacher distribution must sum to one on legal candidates")
    scaled = student_logits / float(temperature)
    log_student = F.log_softmax(scaled.masked_fill(~mask, float("-inf")), dim=-1)
    # xlogy gives the correct zero contribution for zero-probability teacher
    # candidates without taking log(0) in a materialized teacher tensor.
    per_candidate = torch.xlogy(teacher, teacher.clamp_min(torch.finfo(teacher.dtype).tiny)) - teacher * log_student
    per_example = per_candidate.masked_fill(~mask, 0.0).sum(dim=-1)
    if example_weights is not None:
        if not isinstance(example_weights, torch.Tensor) or example_weights.shape != per_example.shape:
            raise ValueError("example_weights must have shape [batch]")
        weights = example_weights.to(device=student_logits.device, dtype=student_logits.dtype)
        if not torch.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
            raise ValueError("example_weights must be finite, nonnegative, and have positive sum")
        loss = (per_example * weights).sum() / weights.sum()
    else:
        loss = per_example.mean()
    return loss * float(temperature) ** 2


def _build_distillation_module():
    from torch import nn

    class RoutingDistillationLoss(nn.Module):
        def __init__(self, *, temperature: float = 1.0):
            super().__init__()
            if (not isinstance(temperature, (int, float)) or not math.isfinite(float(temperature))
                    or not 0.0 < float(temperature)):
                raise ValueError("temperature must be positive")
            self.temperature = float(temperature)

        def forward(self, student_logits, teacher_probabilities, candidate_mask=None, example_weights=None):
            return routing_distillation_loss(
                student_logits,
                teacher_probabilities,
                candidate_mask=candidate_mask,
                example_weights=example_weights,
                temperature=self.temperature,
            )

    return RoutingDistillationLoss


try:  # pragma: no cover - only exercised by a documentation-only environment.
    RoutingDistillationLoss = _build_distillation_module()
except ModuleNotFoundError:  # pragma: no cover
    RoutingDistillationLoss = None
