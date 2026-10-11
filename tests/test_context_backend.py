"""Contract tests for causal serialization and the Canopy-Jev adapter."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_routing.backend import CanopyBackend, validate_answers, validate_questions
from jev_routing.schema import Expert, RoutingContext, TokenState


class _FakeCanopy:
    """A protocol fixture. It never represents a Canopy-Jev inference result."""

    def __init__(self):
        self.requests: list[dict] = []

    def predict(self, *, state, questions, template):
        self.requests.append({"state": state, "questions": questions, "template": template})
        rows = []
        for question in questions:
            keys = list(question["criteria"])
            rows.append({"id": str(question["id"]),
                         "probabilities": {key: 1.0 / len(keys) for key in keys}})
        return {"questions": rows, "computation": {"fixture": True}}


class _QuestionLimitedCanopyBackend(CanopyBackend):
    def __init__(self, model, limit: int):
        super().__init__(model)
        self.limit = limit

    def _fits(self, state, questions):
        return len(questions) <= self.limit


def _context(mode="causal"):
    return RoutingContext(
        tokens=("A", "router", "secret-future", "answer"),
        targets=(TokenState(1, observation={"available": "now"}),
                 TokenState(3, observation={"available": "later"})),
        experts=(Expert("math", "Solves arithmetic", capacity=4),),
        task="Route each token using only observations available at that token.",
        mode=mode,
        history=(
            {"visible_after": 1, "event": "already-observed"},
            {"visible_after": 2, "event": "future-event"},
        ),
        metadata=(
            {"visible_after": 1, "domain": "toy"},
            {"visible_after": 4, "label": "not-yet-visible"},
        ),
    )


def _question(identifier: str) -> dict:
    return {
        "id": identifier,
        "type": "choice",
        "question": "Which permitted action has larger estimated value?",
        "criteria": {"skip": "Do no extra work", "compute": "Use an expert"},
    }


class ContextContractTests(unittest.TestCase):
    def test_causal_target_receives_strict_legal_prefix_and_stamped_history(self):
        state = json.loads(_context().for_target(1))
        self.assertEqual(state["mode"], "causal")
        self.assertEqual(state["visible_through"], 1)
        self.assertEqual([row["position"] for row in state["tokens"]], [0, 1])
        self.assertNotIn("secret-future", json.dumps(state))
        self.assertEqual([row["position"] for row in state["visible_states"]], [1])
        self.assertEqual([row["event"] for row in state["history"]], ["already-observed"])
        self.assertEqual([row["domain"] for row in state["metadata"]], ["toy"])

    def test_bidirectional_mode_explicitly_includes_the_full_token_context(self):
        state = json.loads(_context("bidirectional").for_target(1))
        self.assertEqual(state["mode"], "bidirectional")
        self.assertEqual(state["visible_through"], 3)
        self.assertEqual([row["position"] for row in state["tokens"]], [0, 1, 2, 3])
        self.assertIn("secret-future", json.dumps(state))
        self.assertEqual([row["position"] for row in state["visible_states"]], [1, 3])
        self.assertEqual([row["event"] for row in state["history"]],
                         ["already-observed", "future-event"])
        self.assertNotIn("not-yet-visible", json.dumps(state))

    def test_target_must_be_declared_in_the_context(self):
        with self.assertRaises(ValueError):
            _context().for_target(TokenState(0))


class CanopyAdapterTests(unittest.TestCase):
    def test_oversized_packed_request_is_split_without_changing_the_state(self):
        model = _FakeCanopy()
        backend = _QuestionLimitedCanopyBackend(model, limit=2)
        state = _context().for_target(1)
        answers = backend.decide(state, [_question("a"), _question("b"), _question("c")])
        self.assertEqual([answer["id"] for answer in answers], ["a", "b", "c"])
        self.assertEqual([[q["id"] for q in request["questions"]] for request in model.requests],
                         [["a", "b"], ["c"]])
        self.assertEqual({request["state"] for request in model.requests}, {state})
        self.assertTrue(all(request["template"] == "zero_shot_choice" for request in model.requests))
        self.assertEqual([receipt["question_ids"] for receipt in backend.receipts], [["a", "b"], ["c"]])

    def test_single_request_that_exceeds_limit_fails_before_any_model_call(self):
        model = _FakeCanopy()
        backend = _QuestionLimitedCanopyBackend(model, limit=0)
        with self.assertRaisesRegex(ValueError, "no tokens were truncated"):
            backend.decide(_context().for_target(1), [_question("too-large")])
        self.assertEqual(model.requests, [])

    def test_candidate_axis_is_limited_to_ten(self):
        question = {"id": "too-many", "type": "choice", "question": "Choose.",
                    "criteria": {str(i): str(i) for i in range(11)}}
        with self.assertRaisesRegex(ValueError, "1..10"):
            validate_questions([question])

    def test_backend_rejects_missing_and_invalid_probabilities(self):
        questions = [_question("route")]
        with self.assertRaisesRegex(ValueError, "wrong candidate axis"):
            validate_answers(questions, [{"id": "route", "probabilities": {"skip": 1.0}}])
        with self.assertRaisesRegex(ValueError, "invalid probabilities"):
            validate_answers(questions, [{"id": "route",
                                         "probabilities": {"skip": float("nan"), "compute": 0.0}}])
        with self.assertRaisesRegex(ValueError, "sum to one"):
            validate_answers(questions, [{"id": "route",
                                         "probabilities": {"skip": 0.8, "compute": 0.8}}])


if __name__ == "__main__":
    unittest.main()
