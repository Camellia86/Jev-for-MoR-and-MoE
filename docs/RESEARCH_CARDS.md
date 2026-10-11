# Research cards: context, computation, and routing

Design date: 2026-10-11. These are falsifiable research proposals, not reported
improvements. The numerical controller is the real Canopy-Jev public model.
Its previous decision benchmarks do not establish that it can route neural
experts or choose useful recursion depths.

The implementation interfaces support investigation. They do not yet supply a
trained expert capability model, a production sparse Transformer integration,
or evidence of end-to-end speed/quality gains. Detailed conditions and
counterexamples are in [ALGORITHMS.md](ALGORITHMS.md); primary sources and
verified implementation locations are in [RELATED_WORK.md](RELATED_WORK.md).

## The underlying difficulty

A token's next computation depends on more than its surface identity: the task,
the evidence it may need, the intermediate work already performed, the actual
capabilities of available experts, and the resources remaining. The difficulty
is making these dependencies actionable without spending more on the routing
decision than on the work it avoids.

Ordinary hidden states can encode all of these dependencies in principle.
Full-context text alone does not expose all of them in practice. The research
question is therefore:

> Under what conditions can an explicit, legal computation state turn semantic
> task understanding into better expert and depth decisions at a competitive
> total cost?

The working hypothesis is that a stable interface between task semantics and
executable operations can support a reusable computation policy. Canopy's shared
state with multiple numerical decisions offers one concrete implementation.
The hypothesis would remain meaningful with a different competent controller.

## Card A — Expert and depth choices as executable contracts

**Principle.** Select a computation by its conditional effect and requirements,
using a contract that joins expert identity, additional recurrence depth,
available evidence, and cost.

**What makes this a substantive hypothesis.** A router can fail because the
chosen representation does not expose distinctions relevant to action choice,
or because its expert IDs do not transfer semantic knowledge about what work
the system can perform. The proposal makes an action's execution semantics an
explicit input. It does not merely enlarge a linear gate.

**Implementation.** JointPlanRouter enumerates small menus of expert ×
additional-step actions. Each has a fixed capability description and cost.
Canopy reads every legal input token, the target's execution observations, and
eligible history. The allocator commits actions online in causal mode and may
coordinate globally in bidirectional mode.

Expert cards for opaque pretrained FFNs must be measured. A hand-written
"mathematics expert" name is not evidence of capability. Freeze the checkpoint,
layer, card-construction data, and intervention method before evaluation.

**Existing principle and proposed increment.** Mixture-of-experts, adaptive
computation, routing networks, and MoDE already establish choosing functions and
skipping work. MoR establishes adaptive recursive depth. MCF-MoE already argues
for richer routing context. The candidate increment is a tested semantic
execution interface used by an actual shared-state numerical teacher, followed
where useful by an inexpensive implementation of the learned policy.

**Minimal mathematics.**

- More informative observations can only improve a Bayes-optimal decision rule
  when action sets and decision costs are held fixed.
- The gain is strict only when the extra information changes the best action.
- The serialized state can remain insufficient even when all text is retained.
- Joint choices have no general expressive advantage over
  \(p(e\mid S)p(k\mid e,S)\). Their usefulness must come from the learned
  objective, supervision, interface, or practical optimization.

**Distinguishing predictions.**

1. Matched prefixes with the same target token but different legally available
   disambiguating evidence should receive different routes *when measured
   action rankings differ*. Route changes alone are not the target.
2. Keeping text fixed while changing actual completed work should alter the
   remaining-depth choice when the best continuation changes.
3. Consistently renaming expert IDs and their fixed cards should preserve the
   decision after remapping. Scrambling cards independently of expert functions
   should reduce outcome quality if the cards are causally useful.
4. Gains should shrink when the baseline already receives a decision-sufficient
   representation, or when all candidate experts have nearly identical effects.

**Decisive validation.** On a bounded set of held-out legal states, execute a
small predeclared candidate menu from the same saved executor state and record
actual later loss/accuracy and cost. Compare selection regret, then end-to-end
quality. Do not show those held-out outcomes to the controller. A useful first
study has enough nonsaturated states to distinguish close alternatives; a
synthetic routing puzzle only checks whether the interface expresses the idea.

**Strongest controls.** Same executor, action menu, cache policy, legal input,
history, budgets, and capability evidence; compare a general contextual router,
a conditional expert-then-depth router, the original router, and the same
Canopy model with the disputed information removed. A comparable large
pretrained teacher controls for adding a 27B decision model.

**Failure interpretation.** If card relabeling is inconsistent or cards fail to
predict interventions, the semantic bridge is inadequate. If useful extra
information exists but Canopy cannot exploit it, this controller/serialization
fails. If a matched generic router exploits it equally well, there is no
evidence for a special JEV advantage. These findings do not imply every possible
contextual routing method fails.

**Next uncertainty to eliminate.** Do measured expert × depth action rankings
contain predictable information beyond a strong current-hidden-state router?

## Card B — Recursion as investment in later computation

**Principle.** Spend compute when it improves a later decision enough to justify
its full cost, including preparation whose value appears only after additional
steps or downstream token use.

**What makes this substantive.** A difficult or uncertain token need not benefit
from more of the available computation. Conversely, a token whose current
prediction is already easy may perform useful work for later tokens. The
controller's object is the consequence of a computation, rather than a generic
importance or uncertainty score.

**Implementation.** UtilityRouter provides explicit signed Score scales.
AdaptiveRecursionRouter repeatedly commits actions and refreshes observations.
These are practical interfaces for a future value model; their current ordinal
scores are proxies. A bounded two-step or complete-plan alternative is needed
to test delayed gains. Calibration or outcome-supervised values require new
executor measurements.

**Existing principle and proposed increment.** Rational metareasoning and
value-of-computation theory are the main inspiration, not a new theorem claimed
here. ACT and PonderNet learn when to stop; MoD already observes that token
selection changes what later tokens can attend to. The open candidate is whether
explicit dependencies, execution history, and semantic action descriptions let
the same teacher estimate useful continuations across expert/depth choices.

**Minimal mathematics.**

\[
 G_\pi(S,a)
 =\mathbb E[\mathcal L_{\rm stop}-\mathcal L_{a\to\pi}\mid S]
  -\lambda\,\mathbb E[C_{a\to\pi}\mid S].
\]

Use affected downstream losses when an action changes reusable memory.
For a sufficient state this leads to a resource-constrained Bellman equation.
The premise that the value is accurately estimated is not established by
writing the equation or asking a Score question.

**Sharp counterexamples.**

- Two necessary steps can each have zero immediate gain but positive combined
  gain. A myopic stopping rule can reject the only useful path.
- Equal output entropy can reflect resolvable structure or irreducible noise.
  Entropy thresholds need not track marginal computation value.
- A source token can matter to a later query despite having low current loss.
  Removing its KV can undo the benefit of selecting the later query.
- A plan that looks useful before executing a step may become unnecessary
  afterward; an open-loop fixed schedule cannot react.

**Distinguishing predictions.**

1. A bounded planner should preferentially improve measured delayed-gain cases,
   not merely allocate more steps everywhere.
2. The value of preparation should fall when later consumers are removed or
   made independent, holding the legal information and work budget fixed.
3. Updated execution history should improve stopping only when it predicts a
   change in remaining action value.
4. The benefit should disappear when controller/planning overhead exceeds the
   saved or useful executor work.

**Decisive validation.** Predefine a small intervention study separating
immediate-gain, delayed-gain, and irreducible-uncertainty states using measured
executor outcomes. Choose the split without inspecting which controller wins.
Compare one-step choices, bounded two-step plans, and the strongest adaptive
baseline at matched total computation. Continue to real task quality after a
positive mechanism result; toy success is not the final claim.

**Strongest controls.** Same state, candidate sequences, number of diagnostic
rollouts, cache access, and total planning/execution budget. Include a
task-trained value model and an ordinary short-horizon planner. Give the
baseline the same possibility of investing in source tokens or preparation.

**Failure interpretation.** Accurate next-step scores with poor long-run
choices support a credit-horizon problem. Poor scores already at one step
support a state/utility-estimation problem. No remaining gain under the
strongest same-information planner limits the proposed controller's value.

**Next uncertainty to eliminate.** Are non-myopic gains common and predictable
enough to pay for the extra planning in the intended tasks?

## Card C — Shared-context macro plans and compiled routing

**Principle.** Expensive semantic interpretation should choose a reusable
computation policy or complete feasible plan, while cheap execution handles
individual expert calls and recurrence steps.

**What makes this substantive.** The best information source for deciding
computation may be too expensive to place in every layer. Sharing a legal
context across related choices and distilling decisions changes the
organization of the computation: expensive task interpretation and inexpensive
execution occur at different frequencies.

**Implementation.** PlanSlateRouter chooses among at most ten complete feasible
plans. Budgets and capacities are satisfied before JEV is called. The candidates
must cover meaningfully different allocations, including the strongest baseline.
A later student can imitate selected plans or action distributions. The current
training utilities do not constitute a trained student.

In causal decoding, a macro policy is chosen from the observed prefix and can
govern future work only without reading future tokens. Global plans for all
observed input tokens use the explicit bidirectional/prefix-processing mode.
The API's isolated question branches do not themselves coordinate shared
resources.

**Existing principle and proposed increment.** Temporal abstraction, policy
distillation, and computation selection are established ideas. Canopy contributes
an actual shared-prefix, isolated-branch numerical inference implementation.
The proposed increment is to test whether context-conditioned plans preserve
useful routing behavior after amortization, rather than assume a 27B teacher
can be inserted into an ordinary MoE's cheap routing budget.

**Minimal mathematics.**

- With utility error at most \(\varepsilon\) over a slate, cardinal-value
  selection regret within that slate is at most \(2\varepsilon\).
- Relative to every feasible plan, add the slate's omitted-utility gap
  \(\delta\). A perfect selector cannot repair a poor proposal generator.
- A necessary amortization condition is
  \(H\Delta C>C_{\rm controller}+C_{\rm added\ overhead}\).
- Imitation loss measures agreement, not improvement in task utility.

**Distinguishing predictions.**

1. The advantage should grow with the number of decisions sharing relevant
   context, until plan staleness or slate omission dominates.
2. Complete-plan selection should improve over independently selected
   per-target choices when resources or dependencies interact; it need not
   improve independent, unconstrained cases.
3. A student should preserve performance best on execution states represented
   in its teacher data, and degrade when executor changes invalidate the cards.
4. If the extra context is only useful for one immediate token and cannot be
   amortized, the direct teacher can improve quality while still losing on
   total latency/FLOPs.

**Decisive validation.** Fix one nontrivial slate generator, executor, and legal
context protocol. Measure choice regret and total cost as plan horizon/group
size varies over a small predeclared set. Test the eventual student on held-out
sequences and on states visited by its own policy. Charge teacher labeling,
calibration, and student training separately from serving.

**Strongest controls.** The same slate with heuristic, small contextual,
outcome-supervised, and comparable pretrained-teacher selectors; a generic
student receiving the same label budget; and direct extra executor computation
spending the amount consumed by the controller.

**Failure interpretation.** Good selection within a weak slate points to
proposal coverage. Strong teacher decisions with a weak student point to
distillation or state-distribution mismatch. Good task quality without positive
cost break-even supports an expensive teacher use case, not efficient serving.

**Next uncertainty to eliminate.** How many genuinely useful routing decisions
can share one legal Canopy state before staleness and dependence invalidate the
plan?

## Selection and bounded progression

| Candidate | Main uncertainty | Earliest decisive measurement | Main risk |
| --- | --- | --- | --- |
| A: executable joint contracts | Whether context and cards predict actual action rankings | Matched-state routing regret against strong contextual controls | Semantic labels do not identify neural expert effects |
| B: value of later computation | Whether delayed gains are predictable and worthwhile | Short-horizon intervention study with total cost | Poor value estimates and expensive planning |
| C: macro plans and compilation | Whether useful decisions can be amortized | Slate regret and break-even curve, then student quality | Slate omission and policy-distribution shift |

The recommended order is A's bounded action-ranking audit, then C's
amortization test using the same measured states. Pursue B when those outcomes
actually reveal delayed or downstream utility. This selects the most important
uncertainty, rather than expanding a grid of router variants.

Before a real experiment, freeze the task/data partitions, executor checkpoint,
action catalogue, capability-card provenance, context visibility, cache policy,
cost units, controller revision, and complete set of strong controls. Keep
functional validation separate from outcome evidence.

If the inherited Loop research quality criterion is used for a comparable
accuracy/NLL task, final confirmation requires three independent training seeds,
mean accuracy at least 0.8 percentage points above the strongest matched control,
and no NLL degradation. Teacher replay, evaluation seeds, repeated scoring, or
three continuations of one already-trained model do not create independent
training evidence. For a different objective, preregister its appropriate
quality and resource criterion before seeing results.
