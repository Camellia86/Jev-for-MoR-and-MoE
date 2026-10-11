"""Independent checks of routing allocation, causality, and complete slates."""
from __future__ import annotations

from itertools import product
from pathlib import Path
import sys
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from jev_routing.allocation import allocate
from jev_routing.backend import ReplayBackend
from jev_routing.routers import JointPlanRouter, SignedGainBin, UtilityRouter
from jev_routing.schema import Action, AllocationOption, Expert, RouteDecision, RoutingContext, TokenState
from jev_routing.slates import PlanSlate, PlanSlateRouter


def _context(*, tokens=("zero", "one", "two", "three"), targets=(1, 3), mode="causal"):
    return RoutingContext(
        tokens=tuple(tokens),
        targets=tuple(TokenState(position, observation={"prefix_marker": position}) for position in targets),
        experts=(Expert("math", "Arithmetic planning", capacity=3),
                 Expert("code", "Code retrieval", capacity=2)),
        task="Route under declared cost and capacity without using a future target.",
        mode=mode,
    )


def _brute_force(options, experts, budget):
    """Small independent enumerator used only to cross-check exact allocation."""
    by_position = {}
    for option in options:
        by_position.setdefault(option.position, []).append(option)
    groups = []
    for position in sorted(by_position):
        offered = by_position[position]
        if not any(option.action.expert_id is None for option in offered):
            offered = [*offered, AllocationOption(position, Action.skip(), 0.0)]
        groups.append(offered)
    capacities = {expert.id: expert.capacity for expert in experts}
    legal = []
    for selected in product(*groups):
        cost = sum(option.action.cost for option in selected)
        steps = {expert.id: 0 for expert in experts}
        for option in selected:
            if option.action.expert_id is not None:
                steps[option.action.expert_id] += option.action.steps
        if cost <= budget and all(capacities[key] is None or steps[key] <= capacities[key]
                                  for key in capacities):
            legal.append((sum(option.value for option in selected), cost, selected))
    # This fixture has a unique maximum.  Lower cost is the documented equal-score tie break.
    return max(legal, key=lambda item: (item[0], -item[1]))


class AllocationTests(unittest.TestCase):
    def test_exact_allocator_matches_small_bruteforce_solution(self):
        experts = (Expert("math", "Arithmetic", capacity=3), Expert("code", "Code", capacity=2))
        m2 = Action("math-two", "math", steps=2, cost=2)
        m1 = Action("math-one", "math", steps=1, cost=1)
        c1 = Action("code-one", "code", steps=1, cost=1)
        options = [
            AllocationOption(0, m2, 5.0), AllocationOption(0, c1, 2.0),
            AllocationOption(1, m2, 6.0), AllocationOption(1, c1, 4.0),
            AllocationOption(2, m1, 3.0), AllocationOption(2, c1, 2.0),
        ]
        expected_score, expected_cost, expected = _brute_force(options, experts, budget=4)
        plan = allocate(options, experts, budget=4)
        self.assertEqual(plan.spent, expected_cost)
        self.assertEqual(sum(decision.score for decision in plan.decisions), expected_score)
        self.assertEqual([decision.action.id for decision in plan.decisions],
                         [option.action.id for option in expected])
        self.assertEqual(plan.expert_calls, {"math": 3, "code": 1})
        self.assertTrue(plan.diagnostics["exact"])

    def test_expert_capacity_counts_steps_not_selected_token_rows(self):
        experts = (Expert("math", "Arithmetic", capacity=2),)
        two_steps = Action("two-step", "math", steps=2, cost=1)
        one_step = Action("one-step", "math", steps=1, cost=1)
        plan = allocate([
            AllocationOption(0, two_steps, 5.0),
            AllocationOption(1, one_step, 4.0),
        ], experts, budget=2)
        self.assertEqual([decision.action.id for decision in plan.decisions], ["two-step", "skip"])
        self.assertEqual(plan.expert_calls, {"math": 2})
        self.assertEqual(plan.spent, 1)

    def test_negative_compute_values_choose_the_zero_work_skip(self):
        experts = (Expert("math", "Arithmetic", capacity=8),)
        plan = allocate([
            AllocationOption(0, Action("bad", "math", steps=1, cost=1), -0.01),
            AllocationOption(1, Action("also-bad", "math", steps=1, cost=1), -5.0),
        ], experts, budget=2)
        self.assertEqual([decision.action.id for decision in plan.decisions], ["skip", "skip"])
        self.assertEqual(plan.spent, 0)
        self.assertEqual(plan.expert_calls, {"math": 0})


class RouterTests(unittest.TestCase):
    def test_utility_router_preserves_negative_gain_as_a_skip(self):
        bins = {str(index): 0.0 for index in range(7)}
        bins["1"] = 1.0  # DEFAULT_SIGNED_GAIN_BINS[1] is negative.
        backend = ReplayBackend({"gain:1:bad": bins})
        action = Action("bad", "math", steps=1, cost=1, description="A costly bad route")
        plan = UtilityRouter(backend, budget=1).route(_context(targets=(1,)), {1: [action]})
        self.assertEqual(plan.decisions[0].action.id, "skip")
        self.assertEqual(plan.spent, 0)
        self.assertIn("not measured task utility", plan.diagnostics["score_semantics"])

    def test_zero_elicited_gain_keeps_the_no_work_baseline(self):
        bins = {str(index): 0.0 for index in range(7)}
        bins["3"] = 1.0  # DEFAULT_SIGNED_GAIN_BINS[3] is exactly zero.
        backend = ReplayBackend({"gain:1:neutral": bins})
        action = Action("neutral", "math", steps=1, cost=1)
        plan = UtilityRouter(backend, budget=1).route(_context(targets=(1,)), {1: [action]})
        self.assertEqual(plan.decisions[0].action.id, "skip")
        self.assertEqual(plan.spent, 0)

    def test_custom_skip_id_is_preserved_for_a_skip_only_target(self):
        defer = Action("defer", None, steps=0, cost=0, description="Do not invoke an expert.")
        backend = ReplayBackend({"joint:1": {"defer": 1.0}})
        plan = JointPlanRouter(backend, budget=0).route(_context(targets=(1,)), {1: [defer]})
        self.assertEqual(plan.decisions[0].action.id, "defer")
        self.assertEqual(plan.spent, 0)

    def test_utility_router_allows_many_independent_score_questions(self):
        """Fixture: 14 actions each use a three-level Score axis, never a 14-way Choice."""
        actions = [Action(f"expert-{index:02d}", "math", steps=1, cost=1)
                   for index in range(14)]
        responses = {
            f"gain:1:{action.id}": {"0": 0.0, "1": 1.0, "2": 0.0}
            for action in actions
        }
        responses["gain:1:expert-13"] = {"0": 0.0, "1": 0.0, "2": 1.0}
        backend = ReplayBackend(responses)
        bins = (
            SignedGainBin("loss", -1.0),
            SignedGainBin("neutral", 0.0),
            SignedGainBin("gain", 2.0),
        )
        plan = UtilityRouter(backend, gain_bins=bins, budget=1).route(
            _context(targets=(1,)), {1: actions})
        self.assertEqual(plan.decisions[0].action.id, "expert-13")
        self.assertEqual(plan.spent, 1)
        self.assertEqual(len(backend.requests[0]["questions"]), 14)
        self.assertTrue(all(len(question["criteria"]) == 3
                            for question in backend.requests[0]["questions"]))

    def test_router_rejects_more_than_ten_route_candidates(self):
        actions = [Action(f"a{i}", "math", steps=1, cost=1) for i in range(10)]
        backend = ReplayBackend({})
        with self.assertRaisesRegex(ValueError, "at most 10"):
            JointPlanRouter(backend, budget=1).route(_context(targets=(1,)), {1: actions})

    def test_causal_early_route_is_invariant_to_future_target_scores_and_tokens(self):
        """A causal route is committed before later targets exist for the router."""
        early = Action("early", "math", steps=1, cost=2)
        later = Action("later", "math", steps=1, cost=2)
        initial = _context(tokens=("zero", "one"), targets=(1,))
        extended = _context(tokens=("zero", "one", "changed-future", "target"), targets=(1, 3))
        probabilities = {
            "joint:1": {"early": 0.51, "skip": 0.49},
            "joint:3": {"later": 0.99, "skip": 0.01},
        }
        early_plan = JointPlanRouter(ReplayBackend(probabilities), budget=2).route(
            initial, {1: [early]})
        extended_plan = JointPlanRouter(ReplayBackend(probabilities), budget=2).route(
            extended, {1: [early], 3: [later]})
        early_decision = next(decision for decision in early_plan.decisions if decision.position == 1)
        extended_decision = next(decision for decision in extended_plan.decisions if decision.position == 1)
        self.assertEqual(early_decision.action.id, "early")
        self.assertEqual(extended_decision.action.id, early_decision.action.id)
        self.assertEqual(extended_plan.spent, 2)
        self.assertEqual(next(decision for decision in extended_plan.decisions if decision.position == 3).action.id,
                         "skip")


class CompletePlanSlateTests(unittest.TestCase):
    def test_causal_mode_refuses_a_multi_token_complete_plan(self):
        context = _context(mode="causal")
        router = PlanSlateRouter(ReplayBackend({}))
        skip = Action.skip()
        slate = PlanSlate("skip", "Skip every target", (
            RouteDecision(1, skip, 0.0), RouteDecision(3, skip, 0.0),
        ))
        with self.assertRaisesRegex(ValueError, "explicit bidirectional"):
            router.route(context, (slate,), budget=0)

    def test_bidirectional_slate_filters_budget_and_capacity_before_choice(self):
        context = _context(mode="bidirectional")
        skip = Action.skip()
        math_two = Action("math-two", "math", steps=2, cost=2, description="Two arithmetic steps")
        within = PlanSlate("within", "Use math at the first target only", (
            RouteDecision(1, math_two, 0.0), RouteDecision(3, skip, 0.0),
        ))
        no_work = PlanSlate("no-work", "Skip both targets", (
            RouteDecision(1, skip, 0.0), RouteDecision(3, skip, 0.0),
        ))
        over_budget = PlanSlate("over-budget", "Spend too much", (
            RouteDecision(1, math_two, 0.0), RouteDecision(3, math_two, 0.0),
        ))
        backend = ReplayBackend({"complete_plan": {"within": 0.8, "no-work": 0.2}})
        plan = PlanSlateRouter(backend).route(context, (within, no_work, over_budget), budget=2)
        self.assertEqual(plan.diagnostics["selected_slate"], "within")
        self.assertEqual(plan.spent, 2)
        criteria = backend.requests[0]["questions"][0]["criteria"]
        self.assertEqual(set(criteria), {"within", "no-work"})


if __name__ == "__main__":
    unittest.main()
