# Jev for MoR and MoE

由于当今JEV paper泛滥，我决定开源[**Jev-for-MoR-and-MoE**](https://github.com/Camellia86/Jev-for-MoR-and-MoE)**，并欢迎任何开发者提交PR！**

**Ruoxi Qiu · 2026 · Research implementation · MIT**

Let a router see the problem, the available computations, and what has already
been computed. This repository uses [Canopy-Jev](https://github.com/Camellia86/Canopy-Jev)
to decide **which tokens receive which experts and how many additional recursive
steps**, under explicit budgets and expert capacities. MoR means *Mixture of
Recursions* here.

Canopy is the actual 27B numerical decision model from that repository. The
production adapter calls its original `predict(state, questions)` API. CPU
replay examples use clearly labelled hand-authored probabilities solely to
exercise routing and execution.

## Implemented algorithms

| Policy | Decision | Budget handling | Research hypothesis |
| --- | --- | --- | --- |
| `JointPlanRouter` | One expert × additional-depth action, including skip, per token | Constrained MAP over Choice preferences | Explicit execution contracts help distinguish computations with similar local token states |
| `UtilityRouter` | Signed gain bins for each candidate action through Canopy Score | Allocate elicited gain proxies under hard integer costs | Allocation should follow downstream value rather than uncertainty alone |
| `AdaptiveRecursionRouter` | Stop or choose another expert after reading an execution receipt | Re-query with remaining resources and current observations | The next useful computation depends on what earlier computation resolved |
| `PlanSlateRouter` | Choose one complete, feasible multi-token plan | Filter complete plans against budget and capacity before Choice | A joint plan can represent dependencies that isolated token decisions miss |

Choice probabilities are preferences. `JointPlanRouter` projects their log
probabilities onto the feasible action set; it does **not** estimate NLL gains.
The Score policy assigns caller-defined signed values to ordered levels; those
values remain heuristics until calibrated on independent execution outcomes.
Complete plan menus are supplied explicitly, with at most ten candidates.

The allocator solves a bounded exact multiple-choice resource DP. Declared costs
are positive integer units for compute and zero for skip; expert capacity counts
*expert steps*, including recursive repetitions. It reports state-space overflow
instead of silently approximating. This solver is a reference implementation,
not a large-scale GPU routing kernel.

## Full context and causal execution

The state includes every legal token and its position, the task, expert capability
cards, current target depth, supplied execution observations, and time-stamped
history/metadata. Raw dense hidden/KV arrays are not automatically serialized.
Capability cards should cite measured skills from fixed training/development
evidence; descriptive expert names do not establish those skills.

In `causal` mode, position *t* reads tokens 0…*t*. Decisions are committed in
position order against the remaining budget; a later token cannot displace an
earlier decision. Global DP and multi-token complete-plan selection are for an
explicit `bidirectional` input, such as a known prompt processed before its
answer or an encoder. Full teacher-forced text is not legal context for earlier
causal-LM positions, even during prefill.

Canopy permits 1–10 candidates per question and at most 16,384 packed tokens.
The backend measures actual Canopy tokenization, splits question batches while
retaining the full state, and rejects an oversized individual decision. No
context is silently truncated. Branch isolation does not remove future text
already present in the shared state.

## Install and run

Lightweight protocol/routing code needs Python 3.10–3.12:

```bash
git clone https://github.com/Camellia86/Jev-for-MoR-and-MoE.git
cd Jev-for-MoR-and-MoE
python -m pip install -e .
python -m unittest discover -s tests -v
python scripts/offline_replay.py
```

The optional sparse residual executor and distillation losses use PyTorch:

```bash
python -m pip install -e '.[torch]'
```

Actual Canopy inference needs the supported upstream Linux/CUDA environment and
**at least 64 GiB free NVIDIA GPU memory**. Install its pinned runtime following
the [upstream instructions](https://github.com/Camellia86/Canopy-Jev#install), then
install this project's pinned package integration:

```bash
python -m pip install -e '.[canopy]'
hf download Camellia86/Canopy-Jev-27B --include 'artifacts/*' --local-dir .
python examples/real_canopy.py --help
```

The base model downloads separately. Its exact revision and the Canopy source
revision are recorded in [the integration notes](docs/CANOPY_INTEGRATION.md).
The real example constructs action questions, receives actual numerical
probabilities and compiles a route plan; bring host-model experts and available
observations to execute those computations in your own MoE/MoR model.

## What to test scientifically

Hold information, total compute and training opportunity fixed. Compare local
routers, a generic router with the identical legal context, fixed/random depth,
matched native MoE/MoR routing, and an equally expensive 27B teacher. Ablate
capability cards and execution receipts without changing the compute budget.
Count Canopy calls, prefix work, observation extraction, planning, dispatch and
host execution in end-to-end cost. A large teacher can improve decisions while
making deployment more expensive.

The repository provides distillation loss utilities for training a cheap student
on Canopy route distributions; outcome calibration is a research extension. No student or
host model has been trained in this release. Unit checks and replay execution
establish implementation behavior, not MoE/MoR quality or speed gains.

[MoE/MoR usage recipes](docs/USAGE.md) ·
[Algorithms and derivations](docs/ALGORITHMS.md) ·
[Research cards and decisive controls](docs/RESEARCH_CARDS.md) ·
[Closest prior work](docs/RELATED_WORK.md) ·
[Validation record](docs/VALIDATION.md)

Shared-backbone depth-control experiment: [中文实验计划](docs/EXPERIMENT_PLAN.zh-CN.md)
and [protocol configuration](docs/EXPERIMENT_CONFIG.v1.json). See the
[mathematical audit](docs/MATHEMATICAL_AUDIT.zh-CN.md) and the
[completed CPU algebra checks](docs/EXPERIMENT_MATH_CHECK.v1.json).
The plan uses RTX 5080 for the pilot and 3×A10 for the formal study; it is a
proposed experiment, with no training or results from this protocol yet.

## 简体中文

核心想法是把路由写成一份可执行的计算计划：**哪些 token 值得继续计算、
由哪些专家计算、追加多少递归深度。** JEV使用本家[Canopy-Jev](https://github.com/Camellia86/Canopy-Jev)实现，
输入完整合法上下文、专家能力卡、当前深度和执行回执，直接得到数值决策。
目前实现了联合动作选择、收益代理预算分配、执行后重路由、完整计划选择四条路线。

自回归模式按位置逐步提交决策；允许全输入的任务才使用全局预算优化。
代码包含真实 Canopy 接口、预算与容量约束、稀疏执行接口、蒸馏损失和可运行例子。
本版本发布的是研究算法实现，还没有真实 MoE/MoR 的训练收益结论。
数学设计由用户指定的 GPT-6-astra、max 推理子代理完成并记录在研究文档中。

## Citation

```bibtex
@software{qiu2026jevcontext,
  author  = {Qiu, Ruoxi},
  title   = {Jev for MoR and MoE},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Camellia86/Jev-for-MoR-and-MoE}
}

@software{qiu2026canopyjev,
  author  = {Qiu, Ruoxi},
  title   = {Canopy-Jev: Shared-State Numerical Decision Inference},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/Camellia86/Canopy-Jev}
}
```

Machine-readable citation: [CITATION.cff](CITATION.cff). Original code is MIT.
Canopy's base-derived weights retain their upstream Apache-2.0 terms and are
downloaded separately. No affiliation is specified.
