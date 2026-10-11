"""CPU checks for sparse execution and provenance-bound router supervision."""
from __future__ import annotations

import json
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

try:
    import torch
except ModuleNotFoundError:  # The core package has no mandatory torch dependency.
    torch = None

from jev_routing.context import context_digest
from jev_routing.execution import CausalCallbackExecutor, SparseMoEExecutor
from jev_routing.routers import AdaptiveRecursionRouter
from jev_routing.schema import Action, Expert, RouteDecision, RoutePlan, RoutingContext, TokenState
from jev_routing.training import RoutingDistillationExample, routing_distillation_loss


def _context():
    return RoutingContext(
        tokens=("prefix", "routed", "future-secret", "later"),
        targets=(TokenState(1, depth=3, observation={"available": "prefix"}),
                 TokenState(3, observation={"available": "later"})),
        experts=(Expert("math", "Bounded arithmetic", capacity=2),
                 Expert("code", "Compact code fact", capacity=1)),
        task="Use only legal prefix state for each selected computation.",
        mode="causal",
    )


class _Add(torch.nn.Module if torch is not None else object):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, value):
        return torch.full_like(value, self.value)


class _ActionAwareBackend:
    """A test double that assigns route mass only from the offered axis."""

    def __init__(self):
        self.requests = []

    def decide(self, state, questions):
        self.requests.append({"state": state, "questions": questions})
        answers = []
        for question in questions:
            keys = list(question["criteria"])
            probabilities = {key: 0.0 for key in keys}
            selected = next((key for key in keys if key != "skip"), "skip")
            probabilities[selected] = 0.8 if selected != "skip" else 1.0
            if selected != "skip":
                probabilities["skip"] = 0.2
            answers.append({"id": str(question["id"]), "probabilities": probabilities})
        return answers


@unittest.skipUnless(torch is not None, "requires optional torch")
class SparseExecutionTests(unittest.TestCase):
    def test_selected_rows_receive_exact_residual_steps_and_unselected_rows_are_unchanged(self):
        math_two = Action("math-two", "math", steps=2, cost=1)
        code_one = Action("code-one", "code", steps=1, cost=1)
        plan = RoutePlan(
            decisions=(
                RouteDecision(0, Action.skip(), 0.0),
                RouteDecision(1, math_two, 1.0),
                RouteDecision(3, code_one, 1.0),
            ),
            spent=2,
            expert_calls={"math": 2, "code": 1},
        )
        hidden = torch.arange(8, dtype=torch.float32).reshape(4, 2)
        executor = SparseMoEExecutor({"math": _Add(1.0), "code": _Add(2.0)},
                                     capacities={"math": 2, "code": 1})
        result = executor.execute(hidden, plan, budget=2, depths={1: 3, 3: 0})
        self.assertTrue(torch.equal(result.hidden[0], hidden[0]))
        self.assertTrue(torch.equal(result.hidden[2], hidden[2]))
        self.assertTrue(torch.equal(result.hidden[1], hidden[1] + 2.0))
        self.assertTrue(torch.equal(result.hidden[3], hidden[3] + 2.0))
        self.assertEqual([(record.position, record.depth) for record in result.records],
                         [(1, 4), (1, 5), (3, 1)])

    def test_executor_rejects_a_ledger_that_counts_rows_instead_of_steps(self):
        action = Action("math-two", "math", steps=2, cost=1)
        plan = RoutePlan((RouteDecision(1, action, 1.0),), spent=1, expert_calls={"math": 1})
        executor = SparseMoEExecutor({"math": _Add(1.0)}, capacities={"math": 2})
        with self.assertRaisesRegex(ValueError, "Action.steps exactly"):
            executor.execute(torch.zeros(2, 1), plan, budget=1)

class CausalCallbackTests(unittest.TestCase):
    def test_causal_callback_receives_only_selected_target_prefix_and_step_depth(self):
        calls = []

        def callback(**kwargs):
            calls.append(kwargs)
            self.assertNotIn("future-secret", kwargs["state"])
            self.assertEqual(kwargs["position"], 1)
            parsed = json.loads(kwargs["state"])
            if kwargs["depth"] == 5:
                self.assertEqual(parsed["target"]["depth"], 4)
                self.assertEqual(parsed["target"]["observation"]["execution_observation"],
                                 {"completed_depth": 4})
            return {"completed_depth": kwargs["depth"]}

        action = Action("math-two", "math", steps=2, cost=1)
        plan = RoutePlan(
            (RouteDecision(1, action, 1.0), RouteDecision(3, Action.skip(), 0.0)),
            spent=1, expert_calls={"math": 2, "code": 0},
        )
        result = CausalCallbackExecutor(callback).execute(_context(), plan)
        self.assertEqual([call["depth"] for call in calls], [4, 5])
        self.assertEqual([(record.position, record.step) for record in result.records], [(1, 1), (1, 2)])
        self.assertEqual(result.observations, {1: {"completed_depth": 5}})


class AdaptiveTests(unittest.TestCase):
    def test_adaptive_second_call_receives_only_current_execution_observation(self):
        backend = _ActionAwareBackend()
        observed_passes = []
        compute = Action("recurse", "math", steps=1, cost=1)

        def offers(current, pass_index):
            observed_passes.append((pass_index, current.targets[0]))
            self.assertEqual(current.tokens, ("prefix", "routed"))
            self.assertEqual(tuple(target.position for target in current.targets), (1,))
            if pass_index == 0:
                return {1: [compute]}
            self.assertEqual(current.targets[0].depth, 4)
            self.assertEqual(current.targets[0].observation["execution_observation"], {"fresh": "receipt"})
            return {1: [Action.skip()]}

        def callback(**kwargs):
            self.assertEqual((kwargs["position"], kwargs["action"].id), (1, "recurse"))
            self.assertNotIn("future-secret", kwargs["state"])
            return {"fresh": "receipt"}

        context = RoutingContext(
            tokens=("prefix", "routed", "future-secret"),
            targets=(TokenState(1, depth=3, observation={"available": "now"}),),
            experts=(Expert("math", "Bounded arithmetic", capacity=2),),
            task="Use current execution receipts only.", mode="causal",
        )
        plan = AdaptiveRecursionRouter(backend, max_passes=2, budget=2).route(
            context, offers, executor=CausalCallbackExecutor(callback))
        self.assertEqual([entry[0] for entry in observed_passes], [0, 1])
        self.assertEqual(plan.spent, 1)
        self.assertEqual(plan.expert_calls, {"math": 1})
        self.assertNotIn("fresh", backend.requests[0]["state"])
        self.assertIn("fresh", backend.requests[1]["state"])

    def test_missing_executor_rejects_selected_adaptive_work(self):
        backend = _ActionAwareBackend()
        context = RoutingContext(
            tokens=("prefix", "routed"), targets=(TokenState(1),),
            experts=(Expert("math", "Bounded arithmetic", capacity=1),),
            task="Require execution for selected work.", mode="causal",
        )
        with self.assertRaisesRegex(ValueError, "no executor"):
            AdaptiveRecursionRouter(backend, max_passes=1, budget=1).route(
                context, {1: [Action("compute", "math", steps=1, cost=1)]})

    def test_bidirectional_stop_is_not_requeried_on_the_next_pass(self):
        class ScriptedBackend:
            def __init__(self):
                self.requests = []

            def decide(self, state, questions):
                self.requests.append({"state": state, "questions": questions})
                rows = []
                for question in questions:
                    keys = list(question["criteria"])
                    probabilities = {key: 0.0 for key in keys}
                    if "defer" in keys:
                        probabilities["defer"] = 1.0
                    elif "compute" in keys:
                        probabilities["compute"], probabilities["skip"] = 0.8, 0.2
                    else:
                        probabilities["skip"] = 1.0
                    rows.append({"id": str(question["id"]), "probabilities": probabilities})
                return rows

        class Executor:
            def __init__(self):
                self.positions = []

            def execute(self, context, plan):
                selected = [decision.position for decision in plan.decisions if decision.action.expert_id]
                self.positions.append(selected)
                return {3: {"executed": True}}

        context = RoutingContext(
            tokens=("zero", "one", "two", "three"),
            targets=(TokenState(1), TokenState(3)),
            experts=(Expert("math", "Bounded arithmetic", capacity=2),),
            task="A complete bidirectional input may stop one target independently.", mode="bidirectional",
        )
        seen_active = []
        defer = Action("defer", None, 0, 0)
        compute = Action("compute", "math", 1, 1)

        def offers(current, pass_index):
            seen_active.append((pass_index, tuple(target.position for target in current.targets)))
            return {1: [defer], 3: [compute] if pass_index == 0 else [Action.skip()]}

        executor = Executor()
        plan = AdaptiveRecursionRouter(ScriptedBackend(), max_passes=2, budget=2).route(
            context, offers, executor=executor)
        self.assertEqual(seen_active, [(0, (1, 3)), (1, (3,))])
        self.assertEqual(executor.positions, [[3]])
        self.assertEqual([decision.action.id for decision in plan.decisions], ["defer", "compute", "skip"])


@unittest.skipUnless(torch is not None, "requires optional torch")
class TrainingContractTests(unittest.TestCase):
    def test_teacher_distribution_is_detached_and_student_receives_gradient(self):
        student = torch.tensor([[0.2, -0.1, 0.3]], requires_grad=True)
        teacher = torch.tensor([[0.1, 0.7, 0.2]], requires_grad=True)
        loss = routing_distillation_loss(student, teacher)
        loss.backward()
        self.assertIsNone(teacher.grad)
        self.assertIsNotNone(student.grad)

    def test_training_rejects_nonfinite_student_logits_on_legal_candidates(self):
        with self.assertRaisesRegex(ValueError, "student logits must be finite"):
            routing_distillation_loss(
                torch.tensor([[float("nan"), 0.0]]), torch.tensor([[0.5, 0.5]])
            )

    def test_distillation_example_binds_causal_target_candidate_axis_and_digest(self):
        context = _context()
        state = context.for_target(1)
        example = RoutingDistillationExample(
            state=state, candidate_ids=("skip", "math-two"),
            teacher_probabilities=(0.2, 0.8), context_digest=context_digest(state),
            target_position=1, mode="causal",
        )
        self.assertEqual(example.target_position, 1)
        with self.assertRaisesRegex(ValueError, "context_digest"):
            RoutingDistillationExample(
                state=state, candidate_ids=("skip", "math-two"),
                teacher_probabilities=(0.2, 0.8), context_digest="0" * 64,
                target_position=1, mode="causal",
            )
        future_state = json.loads(state)
        future_state["visible_through"] = 2
        future_state["tokens"].append({"position": 2, "text": "future-secret"})
        future_state = json.dumps(future_state, sort_keys=True, separators=(",", ":"))
        with self.assertRaisesRegex(ValueError, "end exactly"):
            RoutingDistillationExample(
                state=future_state, candidate_ids=("skip", "math-two"),
                teacher_probabilities=(0.2, 0.8), context_digest=context_digest(future_state),
                target_position=1, mode="causal",
            )


if __name__ == "__main__":
    unittest.main()
