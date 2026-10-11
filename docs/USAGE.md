# Mapping the policies to MoE and MoR

The router compiles a host computation plan. The host registers actual expert or
block implementations and supplies live observations. Costs below are example
integer work units; replace them with a fixed, measured cost convention.

## MoE: choose a registered expert

```python
from jev_routing import Action, JointPlanRouter

# context.experts contains fixed capability cards for the host's actual experts.
# One active expert is selected for each target, or the target skips the FFN.
actions = tuple(
    Action(e.id, e.id, steps=1, cost=1,
           description="Apply this registered FFN once as a residual update.")
    for e in context.experts
)
plan = JointPlanRouter(backend).route(context, actions, budget=32)
```

Choice permits nine compute alternatives plus skip. For larger banks, use an
explicit measured shortlist or `UtilityRouter`, whose individual Score
questions each have at most ten gain levels. More expert questions cost more
Canopy work; the backend can split them without deleting legal context.

To activate multiple experts simultaneously, register a composite expert with
an explicit combine rule, measured cost, and constituent-resource ledger. The
current Action schema names one registered expert. It does not silently
interpret a single ID as an arbitrary top-k set.

## MoR: choose additional recursion depth

```python
from jev_routing import Action, JointPlanRouter

# "shared_block" must be a real recurrence-safe host computation.
actions = tuple(
    Action(f"recur-{steps}", "shared_block", steps=steps, cost=steps,
           description=f"Run the shared block for {steps} additional recursions.")
    for steps in (1, 2, 4)
)
plan = JointPlanRouter(backend).route(context, actions, budget=32)
```

With multiple registered experts, enumerate `(expert, additional_steps)` pairs
to use the same contract for a hybrid. Current depth is in TokenState. A real
attention-containing MoR block must preserve the declared KV policy, causal
order and cache versions. The provided sparse tensor executor is a residual
expert primitive; repeating an FFN does not implement an entire Transformer.

## Re-query after actual execution

```python
from jev_routing import Action, AdaptiveRecursionRouter
from jev_routing.execution import CausalCallbackExecutor

def host_step(*, state, position, action, depth):
    # This function operates on the host's live state and legal memory.
    # Perform the selected expert step, then return JSON observations.
    return host.apply_and_observe(position, action.expert_id, depth)

executor = CausalCallbackExecutor(host_step)
actions = (Action("next", "shared_block", 1, 1, "Execute one more shared-block step."),)
plan = AdaptiveRecursionRouter(backend, max_passes=4).route(
    context, actions, executor=executor, budget=32
)
```

The callback receives a serialized legal prefix, action and depth. It must
really perform the computation before returning a receipt. A skip ends that
target's recursion. The returned adaptive plan records work already executed;
do not dispatch it a second time. For sparse tensors, dispatch one pass at a
time using `SparseMoEExecutor` and maintain the host observations yourself.

## Choose an entire plan

```python
from jev_routing import Action, RouteDecision, PlanSlate, PlanSlateRouter

# Multi-token input plans require context.mode == "bidirectional".
slates = (
    PlanSlate("keep", "Keep all current states.", tuple(
        RouteDecision(t.position, Action.skip(), 0.0) for t in context.targets
    )),
    PlanSlate("baseline", "Use the frozen baseline allocation.", baseline_decisions),
    PlanSlate("dependency", "Spend work on the declared dependency chain.", dependency_decisions),
)
plan = PlanSlateRouter(backend).route(context, slates, budget=32)
```

Every supplied candidate must specify each target exactly once, including
skips. Infeasible plans are removed before asking Canopy. Supply meaningful
alternative plans; a perfect selector cannot recover an omitted plan.

The production backend is `CanopyBackend.from_pretrained(...)`. See
[`examples/real_canopy.py`](../examples/real_canopy.py) for the actual loader and
[`scripts/offline_replay.py`](../scripts/offline_replay.py) for the CPU fixture.
