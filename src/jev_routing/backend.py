"""Use the actual Canopy-Jev public API; the CPU replay path is only a fixture."""
from __future__ import annotations
import copy
import math
from typing import Protocol
from .context import context_digest

CANOPY_REVISION = "b71f97b3bffd28c90306bfea607d2089d75e7d43"
BASE_REVISION = "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0"


class DecisionBackend(Protocol):
    def decide(self, state: str, questions: list[dict]) -> list[dict]: ...


def question_keys(question: dict) -> list[str]:
    kind = question.get("type", "choice")
    if kind == "choice":
        if "criteria" in question:
            return list(question["criteria"])
        return question.get("keys", [str(i) for i in range(len(question.get("options", [])))])
    if kind == "noul":
        return ["no", "yes"]
    if kind == "score":
        return [str(i) for i in range(len(question.get("criteria", question.get("levels", []))))]
    raise ValueError("Unsupported question type")


def validate_questions(questions: list[dict]) -> None:
    if not questions or len({str(q.get("id", i)) for i, q in enumerate(questions)}) != len(questions):
        raise ValueError("Questions need unique IDs and must be nonempty")
    for question in questions:
        keys = question_keys(question)
        if not 1 <= len(keys) <= 10 or len(set(keys)) != len(keys):
            raise ValueError("Canopy requires 1..10 distinct candidates per question")
        if not all(isinstance(k, str) and k for k in keys):
            raise ValueError("Candidate keys must be nonempty strings")
        instruction = question.get("question", question.get("instructions", ""))
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("Questions require instructions")


def validate_answers(questions: list[dict], answers: list[dict]) -> list[dict]:
    expected_ids = [str(q.get("id", i)) for i, q in enumerate(questions)]
    if len(answers) != len(questions) or len({a.get("id") for a in answers}) != len(answers):
        raise ValueError("Canopy returned missing or duplicated questions")
    indexed = {str(a.get("id")): a for a in answers}
    if set(indexed) != set(expected_ids):
        raise ValueError("Canopy returned unexpected question IDs")
    result = []
    for q, identifier in zip(questions, expected_ids):
        answer = indexed[identifier]
        probabilities = answer.get("probabilities", {})
        if set(probabilities) != set(question_keys(q)):
            raise ValueError("Canopy returned the wrong candidate axis")
        values = list(probabilities.values())
        if not all(type(v) in (int, float) and math.isfinite(v) and 0 <= v <= 1 for v in values):
            raise ValueError("Canopy returned invalid probabilities")
        if not math.isclose(sum(values), 1.0, abs_tol=2e-5):
            raise ValueError("Canopy probabilities do not sum to one")
        result.append(copy.deepcopy(answer))
    return result


class CanopyBackend:
    """Retain the complete state while splitting oversized question batches.

    Real Canopy tokenization measures the exact shared-prefix tree. A single
    oversized state+question fails visibly: no context truncation is performed.
    """
    def __init__(self, model, *, max_questions: int = 32):
        if type(max_questions) is not int or max_questions < 1:
            raise ValueError("max_questions must be positive")
        self.model = model
        self.max_questions = max_questions
        self.receipts: list[dict] = []

    @classmethod
    def from_pretrained(cls, artifact_dir, *, base_model="Qwen/Qwen3.8-27B",
                        revision=BASE_REVISION, device="cuda:0", local_files_only=False,
                        max_questions=32):
        from canopy_jev import CanopyJev
        model = CanopyJev.from_pretrained(artifact_dir, base_model=base_model,
                    revision=revision, device=device, local_files_only=local_files_only)
        return cls(model, max_questions=max_questions)

    def _fits(self, state: str, questions: list[dict]) -> bool:
        if len(questions) > self.max_questions:
            return False
        if not hasattr(self.model, "tokenizer"):
            # Injected protocol-test doubles have no tokenizer; real Canopy does.
            return True
        from canopy_jev.prompt import prepare
        _, prefix, branches = prepare(self.model.tokenizer, state, questions)
        return len(prefix) + sum(map(len, branches)) <= self.model.max_tokens

    def decide(self, state: str, questions: list[dict]) -> list[dict]:
        questions = [dict(q, id=str(q.get("id", i))) for i, q in enumerate(questions)]
        validate_questions(questions)
        if not isinstance(state, str) or not state:
            raise ValueError("Provide a nonempty serialized context")
        batches, batch = [], []
        for question in questions:
            if not self._fits(state, [question]):
                raise ValueError("Full legal context plus one decision exceeds Canopy's packed-token limit. Supply a smaller explicit problem; no tokens were truncated.")
            if batch and not self._fits(state, [*batch, question]):
                batches.append(batch)
                batch = []
            batch.append(question)
        if batch:
            batches.append(batch)
        answers = []
        for group in batches:
            response = self.model.predict(state=state, questions=group, template="zero_shot_choice")
            answers.extend(validate_answers(group, response["questions"]))
            self.receipts.append({"context_sha256": context_digest(state),
                "question_ids": [str(q.get("id", i)) for i, q in enumerate(group)],
                "computation": copy.deepcopy(response.get("computation", {})),
                "backend": "actual-canopy-jev"})
        return validate_answers(questions, answers)


class ReplayBackend:
    """Hand-authored probability fixtures, not a model or a Canopy substitute."""
    def __init__(self, responses: dict[str, dict[str, float]]):
        self.responses = copy.deepcopy(responses)
        self.requests: list[dict] = []

    def decide(self, state: str, questions: list[dict]) -> list[dict]:
        validate_questions(questions)
        self.requests.append({"state": state, "questions": copy.deepcopy(questions)})
        answers = [{"id": str(q.get("id", i)), "probabilities": self.responses[str(q.get("id", i))]}
                   for i, q in enumerate(questions)]
        return validate_answers(questions, answers)
