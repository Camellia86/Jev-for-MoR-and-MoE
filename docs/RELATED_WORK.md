# Related work and source audit

Sources checked on 2026-10-11. The eight-year boundary used for older conceptual
inspiration is 2018 or earlier. This is a focused nearest-prior audit, not a
claim of exhaustive novelty. Source code was read; the cited models and papers'
experiments were not reproduced.

## Positioning that survives the nearest precedents

Adaptive stopping, recurrent depth, function selection, expert/no-op routing,
context-enriched routing, value of computation, and teacher distillation all
have prior work. The proposed study asks whether an explicit execution contract
and complete legal input let the actual Canopy-Jev numerical controller choose
useful expert/depth plans, and whether those choices can be amortized into a
competitive execution policy.

The research claim needs outcome evidence against same-information controls.
"First joint expert and depth routing", "first context-aware router", or "first
computation-value controller" would not be supported by the sources below.

## Older principles used as inspiration

| Primary source | Original principle | Earlier constraint and present opportunity | What still needs to be new here |
| --- | --- | --- | --- |
| Sutton, Precup, Singh, [Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning](https://www.sciencedirect.com/science/article/pii/S0004370299000521), 1999 | Choose temporally extended actions with an internal policy and termination | Constructing useful abstractions remains hard; a semantic teacher could propose or select bounded computation plans | A demonstrated interface and transfer/cost advantage, not renaming plans as options |
| Hay, Russell, Tolpin, Shimony, [Selecting Computations: Theory and Applications](https://arxiv.org/abs/1207.5879), 2012 | Choose internal computations according to their expected effect on the eventual decision, accounting for cost | Exact metalevel planning can itself be expensive; an amortized predictor offers a practical approximation | Reliable value estimation for actual neural operations under explicit information and budget limits |
| Graves, [Adaptive Computation Time for Recurrent Neural Networks](https://arxiv.org/abs/1603.08983), 2016 | Learn how many internal recurrent steps to take, with a cost for prolonged computation | The halting mechanism is tied to a trained recurrent model; the present question adds described expert choices and an external teacher | Evidence that the richer action/state interface adds value beyond a strong learned halting rule |
| Rosenbaum, Klinger, Riemer, [Routing Networks: Adaptive Selection of Non-linear Functions for Multi-Task Learning](https://arxiv.org/abs/1711.01239), 2017/ICLR 2018 | A router composes function blocks adaptively | Learning useful compositions and assigning their credit are central difficulties; current pretrained controllers bring semantic knowledge | Grounded expert semantics, causal execution, and measured benefits rather than function selection itself |
| Shazeer et al., [Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538), 2017 | Activate a sparse learned combination of experts | Routing and load balance must work with real hardware; this constraint persists | Any gain must survive dispatch and complete controller-cost accounting |
| Dehghani et al., [Universal Transformers](https://arxiv.org/abs/1807.03819), 2018 | Reuse a recurrent transformation over depth, with adaptive per-position computation | Adaptive recurrent representations already exist; present systems additionally expose different experts and explicit cache policies | A better decision interface and practical evidence, not recurrence or adaptive depth alone |
| Rusu et al., [Policy Distillation](https://arxiv.org/abs/1511.06295), 2015/2016 | Train a smaller policy from a stronger teacher | Compact policy imitation is established; Canopy provides a concrete expensive routing teacher | Preservation of useful routes and a positive total-cost tradeoff on new tasks |
| Ross, Gordon, Bagnell, [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning](https://proceedings.mlr.press/v15/ross11a.html), 2011 | Account for the state distribution induced by the learner's own decisions | Offline teacher traces need not cover student-visited states | The state-distribution problem must be measured; a distillation file alone does not solve it |

The main conceptual import is to treat compute as an action with consequences
and termination, while recognizing that choosing the action has its own cost.

## Closest modern routing work

### Switch Transformers

Fedus, Zoph, and Shazeer's
[Switch Transformers](https://arxiv.org/abs/2101.03961), 2021 preprint/JMLR 2022,
uses sparse top-1 expert routing and addresses capacity, load balancing, and
numerical training stability. The source implementation confirms that expert
selection and capacity enforcement are distinct operations.

This is a basic strong control for expert choice, not evidence that each expert
has a human-readable skill. Its hidden input is contextual. The proposed
Canopy controller adds explicit state and capability descriptions and incurs
very different inference cost.

### Mixture-of-Depths and MoDE

Raposo et al.,
[Mixture-of-Depths: Dynamically allocating compute in transformer-based language models](https://arxiv.org/html/2404.02258v1),
2024, allocates a fixed number of tokens to each expensive block. Section 3.5
explicitly discusses the noncausality of sequence-wide top-k selection and
predictor-based autoregressive sampling.

Section 4.3 already includes Mixture-of-Depths-and-Experts: staged depth then
expert routing, and an integrated expert/no-op router. Section 5 discusses how
selection affects future attention and possible query/KV separation. Thus
combining expert and skip decisions or noticing downstream memory value is
not sufficient novelty for this project.

### Mixture-of-Recursions

Bae et al.,
[Mixture-of-Recursions: Learning Dynamic Recursive Depths for Adaptive Token-Level Computation](https://arxiv.org/html/2507.10524v1),
2025, is the direct recurrent-depth baseline. Its expert-choice variant filters
tokens at each recursion; its token-choice variant chooses a depth from the
initial state. The paper compares recursion-specific KV with sharing the
first-recursion KV across later steps.

These mechanisms already cover adaptive token depth, parameter sharing, and
cache design. The present extension would condition computation decisions on
explicit task/executor contracts and a shared-state semantic teacher. It must
compare against a properly configured MoR router with the same legal information
and cache access. Reported MoR quality and throughput are the authors' results;
this repository has not reproduced them.

### PonderNet and Expert Choice

Banino, Balaguer, and Blundell's
[PonderNet: Learning to Ponder](https://arxiv.org/abs/2107.05407), 2021, is a
close learned-stopping reference. A history-aware recursion controller should
not be compared only with fixed depth.

Zhou et al.,
[Mixture-of-Experts with Expert Choice Routing](https://arxiv.org/abs/2202.09368),
2022, chooses tokens per expert under fixed expert buckets. It is relevant to
the distinction between per-token choice and capacity allocation. A globally
balanced selection is not automatically a legal causal decision for each
position.

### Multi-level Context Fusion MoE

Huang et al.,
[Multi-level context Modeling for consistent expert selection in Mixture-of-Experts](https://arxiv.org/abs/2607.16427v2),
2026, directly frames routing as a contextual-completeness problem. The paper and
[author repository](https://github.com/shuhanhuang/MCF-MOE/tree/e24b73c1f2ccb06e184c8de89ebb57fc7a398362)
describe local token context plus cross-layer context for expert selection.
This is a particularly close precedent to the broad motivation here.

The distinction under test is therefore not "give the router context".
It is the execution contract, semantic teacher interface, explicit legal
visibility, and macro-plan/teacher-to-student organization. Full core-code
verification was limited by the missing imported modules described below.

## Verified implementation evidence

The commits below were obtained from public source metadata and the relevant
files were read. Pinning a current source snapshot does not establish that it
is exactly the version used for the paper's reported experiments.

| Project and snapshot | Inspected location | Code observation | Interpretation and boundary |
| --- | --- | --- | --- |
| Canopy-Jev, b71f97b3bffd28c90306bfea607d2089d75e7d43 | [prompt.py](https://github.com/Camellia86/Canopy-Jev/blob/b71f97b3bffd28c90306bfea607d2089d75e7d43/src/canopy_jev/prompt.py), clean_questions, zero_shot_prompt, common_prefix, prepare | Explicit 1–10 candidate adapters; exact shared prefix; separate question branches | Actual numerical decision interface; no invented token/variable embedding replacement |
| Canopy-Jev, same snapshot | [model.py](https://github.com/Camellia86/Canopy-Jev/blob/b71f97b3bffd28c90306bfea607d2089d75e7d43/src/canopy_jev/model.py), validate_geometry, infer_tokens, predict | Packed-token limit, fresh tree inference, frozen inference path, output probabilities, zero generated answers | Confirms integration contract; no persistent cross-request cache or task-loss gradient is exposed by this API |
| Switch, tensorflow/mesh e6798a2610a2c2f4c4cd236d8214422cb1ecc00a | [moe.py, _switch_gating](https://github.com/tensorflow/mesh/blob/e6798a2610a2c2f4c4cd236d8214422cb1ecc00a/mesh_tensorflow/transformer/moe.py#L1303) and transformer_moe_layer_v1 | FP32 gate computation, dense logits, softmax/top-1 or sampling, density-based auxiliary loss, cumulative capacity positions, masked overflow, separate dispatch/combine tensors | Capacity can change the executed path after the preferred expert is chosen; comparing preference alone misses execution effects |
| MoR, 82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6 | [expert_choice_router.py](https://github.com/raymin0223/mixture_of_recursions/blob/82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6/model/mor_model/expert_choice_router.py), MoRLlamaDecoderLayer.forward | Scores gathered active tokens, selects top-k over token positions, computes blocks, then weighted scatter-add; auxiliary targets are formed during training | Selection itself remains sequence-dependent in this inspected path; a trained auxiliary loss does not by itself prove causal incremental execution |
| MoR, same snapshot | [token_choice_router.py](https://github.com/raymin0223/mixture_of_recursions/blob/82c97f29cd140efa01f5cc4fe0de73a4c3b6ffd6/model/mor_model/token_choice_router.py), forward and select_tokens_and_batch_with_padding | Predicts recurrence class from the input, selects depth top-1, gathers still-active tokens with padding, combines completed states, and computes balancing diagnostics/loss | Direct control for up-front depth choice; padding and KV policy matter to actual work |
| MoR, same snapshot | Both routing files' kv_sharing branches | The inspected paths run full-token block inputs for KV sharing, then extract selected results | Sparse abstract query counts are not sufficient to infer the FLOPs or latency of these code paths |
| MCF-MOE, e24b73c1f2ccb06e184c8de89ebb57fc7a398362 | [custom_glb_gate.py](https://github.com/shuhanhuang/MCF-MOE/blob/e24b73c1f2ccb06e184c8de89ebb57fc7a398362/code/custom_glb_gate.py), public recursive file tree | Imports global_router_gate, local_neighborhood_router_gate, and ablation_global_router_gate; those Python modules were absent from the inspected repository tree | Paper/README motivation verified; full central routing implementation was not reconstructible from this snapshot |
| MoD, paper v1 | Sections 3.3–3.5 and 4.3 | Mathematical routing and training/sampling protocols read | No official executable source was acquired for this audit; no reproduction claim |

For MoR, the inspected expert-choice forward calls token top-k outside a
training-only conditional and contains an unimplemented 2D inference-mask path.
This is a specific observation about that file. It does not establish that
every other inference path in the repository is absent or incorrect. A
production baseline needs a separately verified causal sampling path.

The Canopy source inspected here uses a frozen 27B base and adapted final
layers, not a cheap per-layer gate. Its
[README](https://github.com/Camellia86/Canopy-Jev/blob/b71f97b3bffd28c90306bfea607d2089d75e7d43/README.md)
reports 272–741 ms P50 over its specified zero-shot H200 workloads and requires
at least 64 GiB free GPU memory for the supported loader. Those are upstream
reported conditions, not measurements of this routing project.

## What the source audit changes about the experiments

1. A richer-context generic router is mandatory: MCF-MoE is a direct conceptual
   precedent, and ordinary hidden states are already contextual.
2. The conditional expert-then-depth control is mandatory: MoDE already combines
   the axes, and factorization does not itself destroy expressivity.
3. Strict causal claims need prefix-invariance tests of the *whole allocator*.
   Per-question causal input does not make global budget projection causal.
4. Recursion-specific and shared-base KV are distinct experimental conditions.
   An advantage from retaining more keys cannot be attributed solely to routing.
5. A 27B teacher's prior knowledge and cost must be controlled. A gain over a
   tiny linear gate would not isolate the proposed execution interface.
6. Macro planning and distillation are candidate deployment strategies. Their
   value must be established by outcome retention and total cost.

The novelty question remains empirical and comparative: do the proposed
contracts expose useful decision information, can the actual Canopy teacher use
it, and can the resulting policy outperform equally informed alternatives
within the relevant resource budget?
