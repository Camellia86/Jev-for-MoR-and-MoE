# Algorithms and conditional guarantees

Research design dated 2026-10-11. The controller is Ruoxi Qiu's actual
[Canopy-Jev](https://github.com/Camellia86/Canopy-Jev/tree/b71f97b3bffd28c90306bfea607d2089d75e7d43),
not a newly invented neural embedding layer. This document separates the
implemented decision interfaces from proposed training and integration. No
Canopy model inference, routing-quality experiment, or student training was
performed to establish the arguments below.

## 1. The question and the computation interface

The research question is whether an explicit description of the available
context, expert capabilities, execution history, and remaining resources lets a
semantic controller allocate useful computation more effectively. A normal MoE
hidden state already contains contextual information. The hypothesis concerns
which information remains usable by a particular router at a particular cost;
it does not assume ordinary routers have no context.

At a legal decision frontier, let:

- \(x_{\leq t}\) be the observed token prefix;
- \(h_i^r\) be target token \(i\)'s state after \(r\) executed recurrence steps;
- \(M\) be the executor's available KV or other working memory;
- \(H\) be the executed-action history;
- \(B\) be the remaining declared work budget;
- \(C_e\) be the remaining capacity of expert \(e\);
- \(D_e\) be a versioned capability description for expert \(e\).

A micro-action is \(a=(e,k)\): apply the registered expert or shared block for
\(k\) additional steps to one target. The skip action preserves its current
state. Its cost \(c(a)\), prerequisites, and cache effects are properties of
the executor, not numbers that JEV is allowed to invent.

The initial schema has one expert ID per Action. An MoE expert set or a
heterogeneous sequence \(e_1,e_2,\ldots\) requires an explicitly registered
composite action or a later schema extension. Repeating an FFN is not by itself
equivalent to repeating an attention-containing MoR block.

A typical underlying operation is

\[
  \widetilde h_i^r
  = h_i^r+\operatorname{Attn}(h_i^r,M_{\leq i}^r),\qquad
  h_i^{r+1}
  = \widetilde h_i^r+\sum_{e\in E_i^r}w_{i,e}^r
    F_e(\widetilde h_i^r).
\]

This is a possible executor contract, not an assertion that this repository
has implemented a production Transformer with this equation. The controller
chooses a registered path. The host performs the tensor computation.

## 2. What the real Canopy API supplies

The verified source path is:

1. [prompt.py, clean_questions/prepare](https://github.com/Camellia86/Canopy-Jev/blob/b71f97b3bffd28c90306bfea607d2089d75e7d43/src/canopy_jev/prompt.py)
   converts a shared state and questions into exact shared-prefix and isolated
   branch token sequences.
2. [model.py, predict/infer_tokens](https://github.com/Camellia86/Canopy-Jev/blob/b71f97b3bffd28c90306bfea607d2089d75e7d43/src/canopy_jev/model.py)
   performs fresh full-model tree inference and returns fixed-policy numerical
   probabilities.
3. Choice supports 1–10 candidates. Noul is the same head with two candidates.
   Score is the same head with ordered candidates. The packed limit is
   \(L_{\rm prefix}+\sum_q L_q\leq16{,}384\); overlength input is rejected.

All questions in one request see the same state. Their branch isolation prevents
one question from reading another question's branch, but does not hide anything
already placed in the common state. Separate calls do not expose a public
persistent prefix-cache API. Splitting a large question batch therefore repeats
the shared-state work.

The wrapper retains the full serialized legal input when splitting questions.
If even one question plus that input exceeds the real tokenizer's limit, it
fails instead of silently truncating. This is an input-retention guarantee.
It is not a guarantee of semantic understanding or a representation of every
element of the executor's hidden state.

## 3. Visibility: causal decoding, prefill, and encoding

The current causal convention is that token position \(i\) has observed
\(x_0,\ldots,x_i\) and contributes to predicting \(x_{i+1}\). For a convention
that predicts \(x_i\), shift the cutoff to \(i-1\).

| Use case | Legal state | Allocation scope |
| --- | --- | --- |
| Standard autoregressive LM, including teacher-forced prefill | Prefix through the target's cutoff | Commit in increasing target order; future scores cannot change an earlier choice |
| One decoding frontier | Complete already-observed prefix and execution state available now | Joint actions for this frontier are possible |
| Encoder or explicitly bidirectional input processing | All observed input tokens | Global allocation over those input targets is possible |
| Prompt processing used only to predict later answer tokens | Entire observed prompt, under a declared prefix/encoder computation contract | Global prompt plans are possible; this is not ordinary per-position causal prefill |
| Offline privileged teacher using later labels or suffixes | Explicitly privileged training information | Never report its routes as causal inference; student inputs must remain causal |

The fact that all tokens are in GPU memory during prefill does not make them
legal inputs for every target. A complete-plan choice for multiple causal
positions has the same problem as a sequence-level top-k router. The current
PlanSlateRouter therefore allows only one target in causal mode.

RoutingContext filters history and metadata by their visible_after timestamps.
The host must also ensure that task instructions, expert cards, capacities, and
every TokenState observation were available at the stated cutoff. For example,
a diagnostic attached to an old token but computed after reading a later
answer is still future information. A timestamp is a declaration of provenance,
not proof that the record's contents obey it.

### Proposition 1: prefix invariance of online allocation

Fix a model, action catalogue, expert cards, initial budgets, and tie-breaking
rule before observing the sequence. Suppose that:

1. the state for target \(i\) is a deterministic function of \(x_{\leq i}\) and
   execution records available by \(i\);
2. its JEV decision uses only that state;
3. the allocator commits target \(i\) using only its current probabilities,
   previously committed actions, and remaining resources;
4. an executor step reads only legal memory and uses the same rule in prefill
   and incremental execution.

Then any two sequences with the same prefix through \(i\) receive the same
committed routes through \(i\), and the same resulting states through \(i\),
assuming deterministic execution.

**Proof.** At the first target the state, feasible set, and decision are equal.
If previous decisions are equal, they leave equal budgets, capacities, and
legal caches. Equal prefixes then give equal current states and probabilities,
so the next committed decision and its executor result are equal. Induction
completes the argument. With random routing, couple the same exogenous random
draws; otherwise the conclusion is equality in distribution.

A changing batch-level capacity or a resource pool that depends on future
requests is an additional input. It must be held fixed for this proposition,
or modeled explicitly as part of the observable system state.

### Counterexample: causal scores followed by a global allocator

There are two positions, one unit of budget, and one compute action of cost 1.
The first position's causal preference for compute over skip is 1. For one
suffix, the second position's preference is 0; for another it is 2. A global
max-sum allocator gives the budget to the first position in the first sequence
and to the second position in the other sequence. The first route changes
although its input prefix is identical.

This failure persists if the preferences are log probabilities, learned scores,
or oracle utilities. Causal serialization alone cannot repair it. Strict
causal mode uses online commitment; global optimization is reserved for a
legally shared frontier or bidirectional mode.

## 4. Execution state and dependency preservation

The serializer exposes token text and positions; target depth and observations;
expert descriptions and capacities; legal execution history; and task metadata.
Useful host-supplied observations can include checkpoint identity, block
identity, prior selected experts, cumulative work, residual changes, cached
depths, and a separately trained intermediate readout.

Neither raw token strings nor norms and confidence values reveal arbitrary
hidden-vector content. Passing a hidden vector as thousands of decimal strings
also gives no reason to expect the frozen Canopy model to understand its
coordinate system. A learned diagnostic projection, calibrated probes, or
paired capability evidence would be an additional method requiring its own
training and cost accounting.

Expert names such as "logic" or "syntax" are hypotheses until supported by
measurements. For a pretrained MoE, an expert card should record its model and
layer version, intervention protocol, number of calibration examples, measured
conditional gains and failures, and execution cost. Cards are built using the
training/calibration partition and then frozen before test predictions.
Assignments can vary by layer and recursion state, so a global semantic label
for expert ID 7 is generally inadequate.

Stopping a token must not silently erase context needed by another target.
The executor must declare one cache policy:

- **Shared base KV:** every observed token produces base KV once; active queries
  at greater depth can read the complete legal base context. This follows a
  known MoR design and preserves coverage, not depth-refined representations.
- **Recursion-specific KV:** only entries actually produced at that depth are
  available. The controller must see that some dependencies would disappear.
- **Explicit recomputation or versioned cache:** missing entries are computed
  on demand and charged. An earlier state cannot be overwritten invisibly
  after it has influenced committed predictions.

For shared base KV with \(n_r\) active queries and \(n\) available keys, the
attention term scales like \(n_r n\), not \(n_r^2\). Claiming the latter while
retaining every key would undercount work.

## 5. Algorithm A: JointPlanRouter

For each target, enumerate at most ten legal actions, including skip:

\[
  \mathcal A_i=\{\operatorname{skip},(e,k),\ldots\}.
\]

The question names the target, the current depth, and the candidate contracts.
Each candidate has an explicit capability description, additional step count,
and cost. Canopy returns a distribution \(p_i(a\mid S_i)\).

Define relative preference

\[
  v_i(a)=\log \max(p_i(a),\epsilon)
         -\log\max(p_i(\operatorname{skip}),\epsilon),\qquad
  v_i(\operatorname{skip})=0.
\]

The numerical floor only defines finite behavior for zero probabilities. With
positive probabilities, maximizing the sum of these relative preferences over
one action per target is equivalent to maximizing the product of their choice
probabilities: the skip term is constant across feasible plans.

For a legally shared frontier, a resource-constrained plan is

\[
\begin{aligned}
  \max_{z}\quad & \sum_{i,a} z_{i,a}v_i(a)\\
  \text{s.t.}\quad&
     \sum_a z_{i,a}=1,\quad z_{i,a}\in\{0,1\},\\
  &\sum_{i,a}z_{i,a}c(a)\leq B,\\
  &\sum_{i,a:\,e(a)=e}z_{i,a}k(a)\leq C_e.
\end{aligned}
\]

This is a constrained product-of-preferences objective. It is not a proof that
the plan minimizes task loss. The conditional independence interpretation is
an allocator model, not an independence property of the target computations.
For example, two actions can each look valuable alone while being redundant
together; summing their individual values then overestimates the pair.
Expert capacities count executed calls/steps under this contract; a deployment
using another unit must change the accounting consistently.

In strict causal mode the same value rule is applied one target at a time in
increasing position order, updating the remaining budget and capacities after
each commitment. This forfeits hindsight-optimal budget allocation to preserve
the specified information boundary. A known, ex-ante reservation policy can
reduce early budget exhaustion; it cannot inspect future scores.

### Joint actions and factorization

Joint actions make feasibility, expert identity, and depth effects explicit.
They do not have a universal expressivity advantage over a properly conditional
factorization:

\[
  p(e,k\mid S)=p(e\mid S)\,p(k\mid e,S).
\]

Even separate deterministic heads can represent the optimal pair if both receive
a sufficiently informative state and are trained appropriately. The appropriate
control therefore includes a conditional expert-then-depth router with the
same information and action budget.

There is a narrower failure of choosing each axis from separately averaged
utilities. For

\[
  U=\begin{pmatrix}5&5\\9&0\end{pmatrix},
\]

uniformly averaging the other axis selects row 1 and column 1, yielding 5,
whereas the best joint action yields 9. This illustrates an unsuitable decision
objective; it does not prove that factorized neural routers are incapable of
learning the correct action.

### Menus larger than ten

The upstream head has at most ten candidates. Do not silently retain the first
ten experts or describe a shortlist as exhaustive. Options are a declared
shortlist, a hierarchical choice with extra calls, or per-action Score questions.
A hierarchy's early errors can remove the best action. Shortlist recall and
construction cost must therefore be measured separately.

## 6. Algorithm B: UtilityRouter and adaptive recursion

UtilityRouter asks Score questions for candidate actions. A fixed signed scale
\(\{u_0,\ldots,u_{m-1}\}\), \(m\leq10\), yields

\[
  \widehat q_i(a)=\sum_{j=0}^{m-1}p_{i,a,j}u_j.
\]

This is an expected **ordinal proxy under an assigned numeric scale**. The
upstream expected_level averages candidate indices; neither quantity is a
measured NLL reduction or a calibrated probability of improvement. Changing
the spacing of ordinal labels changes the expected proxy and can change the
allocation. The prototype makes the scale explicit.

An adaptive controller can execute a small committed action, obtain updated
host observations, and ask again until skip, a recurrence cap, or budget
exhaustion. AdaptiveRecursionRouter expresses this interaction. It must record
actual executed work, rather than counting only the number of controller calls.
The current adaptive implementation reuses JointPlanRouter's relative Choice
preferences; it is not a Bellman solver or an outcome-calibrated utility model.
Its history is useful only when the host really refreshes the executor state.
Repeating a question against unchanged observations is not progress evidence.

### The value-of-computation target

For a fixed executor, define the future task loss after committing to a policy
\(\pi\) as \(\mathcal L_\pi\). Labels may supply training losses but never enter
the inference state. A true action value relative to stopping is

\[
  G_\pi(S,a)
  =\mathbb E[\mathcal L_{\rm stop}-\mathcal L_{a\to\pi}\mid S]
   -\lambda\,\mathbb E[C_{a\to\pi}\mid S].
\]

The expectation is over unknown targets and any policy randomness. Cost includes
controller and executor work. If \(S\) is a sufficient Markov state, the
loss-minimizing Bellman equation is

\[
  V(S,B)=
  \min\left\{
    R_{\rm stop}(S),\
    \min_{\substack{a\ne{\rm stop}\\c(a)\leq B}}
      \left[\lambda c(a)+
       \mathbb E[V(T(S,a),B-c(a))\mid S]\right]
  \right\}.
\]

If the state omits relevant history, use a belief/history state or accept a
partially observed approximation. Canopy's general decision probabilities do
not satisfy this equation merely because an option describes "high utility".
Learning or calibrating these values requires actual executor outcomes.

### Why one-step stopping can fail

Suppose two steps are complementary: neither improves the current output, but
executing both yields utility 1. Each costs 0.2. A one-step gain test rejects
both; a two-step plan has net utility 0.6 and should be accepted. Conversely,
arbitrarily deep lookahead is unaffordable and inaccurate transition predictions
can make it worse.

High entropy also does not identify useful additional computation. Two states
can have the same predictive entropy, while one contains resolvable ambiguity
and the other contains irreducible target randomness. The useful quantity is
the effect of a feasible computation on future loss, not uncertainty alone.

The decisive comparison is a fixed short-horizon action slate versus a myopic
controller, on states where delayed gains can be measured, with the cost of
planning and any diagnostic rollouts included.

## 7. Algorithm C: PlanSlateRouter and amortized control

Construct a small set of complete feasible plans
\(\mathcal P(S)=\{P_1,\ldots,P_m\}\), \(m\leq10\), before querying Canopy.
A plan specifies every target's action, aggregate cost, expert calls,
recurrence endpoints, and any declared cache/dependency policy. Canopy answers
one Choice question over these plans.

This moves budget coordination inside the selected object. Independent question
branches cannot jointly negotiate resources, whereas each complete candidate
already satisfies the shared constraints. It is appropriate for bidirectional
input groups or one legal causal target/frontier. It must not retroactively
choose routes for earlier causal predictions from a later shared state.

The proposal generator matters. Include a skip plan, the strongest established
baseline plan, and a few meaningfully different feasible alternatives. Avoid
filling the slate with cosmetic variants of the intended winner. The catalogue,
candidate ordering, and generator version must be logged.

### Proposition 2: value error and slate omission

Let \(U(P)\) be the true expected utility of a feasible plan and suppose a
cardinal predictor obeys
\(\lvert\widehat U(P)-U(P)\rvert\leq\varepsilon\) for every plan in the slate.
If \(\widehat P=\arg\max_{P\in\mathcal P}\widehat U(P)\), then

\[
  \max_{P\in\mathcal P}U(P)-U(\widehat P)\leq2\varepsilon.
\]

**Proof.** For the best slate plan \(P^*\), insert the predicted values:
\(U(P^*)-U(\widehat P)\leq
[U(P^*)-\widehat U(P^*)]+
[\widehat U(P^*)-\widehat U(\widehat P)]+
[\widehat U(\widehat P)-U(\widehat P)]\leq2\varepsilon\).

Relative to all feasible plans, add the omission gap
\(\delta(S)=\max_{\rm feasible}U-\max_{\mathcal P}U\). Total regret is at most
\(\delta(S)+2\varepsilon\).

This is a conditional approximation result. Choice probabilities and raw Score
levels are not cardinal utility estimates, so the bound does not automatically
apply to this implementation.

### Why macro decisions may be necessary

Let one Canopy call cost \(C_J\), and let each of \(H\) amortized decisions save
\(\Delta C\) executor work on average. Ignoring other costs, a necessary
break-even condition is

\[
  H\Delta C>C_J.
\]

With serialization, scheduling, student work, and additional cache traffic,
replace \(C_J\) by their sum. This is a cost identity, not a prediction that the
inequality will hold. A 27B call for each target at each layer is unlikely to be
an economical replacement for a small matrix router without substantial
amortization.

One deployable research path is offline teacher decisions followed by a cheap
student. Another is infrequent replanning at an observed prompt/chunk boundary.
Future token contents cannot be supplied to make a plan more accurate. A plan
for future decoding positions can use only policies and constraints fixed from
the present prefix.

## 8. Conditional information argument

Let \(C\) be the complete legal observable context, \(Z=f(C)\) a compressed
router input, and \(U(a,Y)\) a bounded action utility. Assume the same action set
and a Bayes-optimal decision rule. Then

\[
\begin{aligned}
  V_C&=\mathbb E[\max_a\mathbb E[U(a,Y)\mid C]],\\
  V_Z&=\mathbb E[\max_a\mathbb E[U(a,Y)\mid Z]],\\
  V_C&\geq V_Z.
\end{aligned}
\]

Condition on \(Z\) and apply the fact that the expectation of a maximum is at
least the maximum of the expectations. The complete-context decision maker
could also emulate any rule based on \(Z\).

Strict gain requires contexts collapsed by \(Z\) for which different actions
are preferable. If \(Z\) is already decision-sufficient, or one action is best
everywhere, the gain is zero. The comparison omits the extra cost of acquiring
and processing \(C\). It gives no guarantee for a fixed pretrained Canopy model
or a finite trained router.

A serialization \(S\) is decision-sufficient only if it preserves the action
values, for example
\(\mathbb E[U(a,Y)\mid C]=\mathbb E[U(a,Y)\mid S]\) for all relevant actions.
Suppose two equally likely hidden execution states have the same serialized
observations, while their optimal experts differ and choosing the wrong expert
loses \(\Delta\). Any policy based only on that serialization suffers expected
regret at least \(\Delta/2\). Providing every token string does not remove this
counterexample when hidden execution details remain omitted.

## 9. Gradient and credit assignment

The public Canopy inference path freezes its model, uses inference mode, and
returns numerical results. Calling predict inside a Transformer does not create
a differentiable path from the language-model loss to Canopy.

Three distinct training choices should remain separate:

1. **Teacher imitation.** Collect actual Canopy distributions on legal states
   and minimize
   \[
     \mathcal L_{\rm KD}
       =\mathbb E_S\,{\rm KL}(p_J(\cdot\mid S)\,\|\,p_\theta(\cdot\mid S)).
   \]
   This teaches the student to imitate the teacher, including its mistakes.
2. **Outcome supervision.** Execute a bounded set of alternatives on training
   states and learn action returns or pairwise preferences. Every extra
   execution is charged to the training budget. Use task losses over affected
   later predictions when the action changes reusable KV.
3. **Task-trained stochastic routing.** For a student policy,
   \[
     \nabla_\theta J
     =\mathbb E[(R-b(S))\nabla_\theta\log\pi_\theta(a\mid S)].
   \]
   The baseline must not depend on the sampled action unless the appropriate
   correction is used. Sequential actions need returns covering their later
   consequences.

Task gradients through selected expert outputs train those experts, and a
continuous selected gate can receive ordinary gradients. The discrete
selection index does not acquire an exact gradient this way. A straight-through
estimator or soft relaxation is a further biased/relaxed training method, not
an automatic property of Choice.

For selected plans projected through a budget solver, either distill the
executed plan or teach the student the same allocator. A distribution before
projection and an executed route after projection are different targets.
Refresh teacher data on student-visited states if distribution shift proves
material; this is proposed future work, not a training result.

## 10. Resource accounting and the strongest controls

Declared Action.cost and Expert.capacity enforce work-unit budgets. They do not
measure total FLOPs or latency. A complete comparison records:

\[
 C_{\rm total}
 =C_{\rm serialize}+C_{\rm Canopy}+C_{\rm diagnostics}
  +C_{\rm allocator}+C_{\rm dispatch}+C_{\rm executor}+C_{\rm cache}.
\]

Also report peak memory, communication, padding, dropped work, and tail latency.
Dropless queued dispatch can preserve the chosen computation while changing
latency. If an overflow instead substitutes or drops an expert, it changes the
executed policy and must be logged.

Prefix sharing saves repeated context encoding inside one Canopy request. It
does not make the 27B trunk free, and zero generated answer tokens does not mean
zero inference computation.

The decisive controls are:

- the executor's original learned router, fixed depth, and a strong tuned
  adaptive-depth router;
- a generic causal/contextual router receiving the same legal information,
  history, action contracts, and resources;
- conditional expert-then-depth factorization with the same inputs;
- the same Canopy model with legal context or execution-history ablated, while
  controlling prompt length/cost where practical;
- a comparably costly pretrained decision teacher with the same state and
  candidates, to isolate Canopy's numerical interface from extra pretrained
  intelligence;
- a student trained from task outcomes or the alternative teacher at matched
  teacher-label and training budgets;
- for plan selection, the same slate chosen by baseline heuristics and measured
  outcome supervision, plus a small exhaustive oracle used only diagnostically.

A functional CPU replay checks bookkeeping. An actual Canopy call checks the
integration. Paired outcome measurements test routing. Independent training
seeds test quality and stability. These are different evidence levels.
