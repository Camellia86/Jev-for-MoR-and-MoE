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

Shared-coda depth-control experiment, separate from the routing package above:
the [中文实验计划](docs/EXPERIMENT_PLAN.zh-CN.md) and
[protocol configuration](docs/EXPERIMENT_CONFIG.v2.json) propose training a
shared two-layer coda LoRA on the frozen Huginn-0125 recurrent model so that one
forward pass both generates text and chooses the recurrence depth. It does not
call Canopy-Jev and does not import `jev_routing`; it borrows Canopy's published
Dirichlet leave-one-out policy-gradient recipe as one training arm. See the
[mathematical audit](docs/MATHEMATICAL_AUDIT.zh-CN.md) and the
[completed CPU algebra checks](docs/EXPERIMENT_MATH_CHECK.v2.json). The plan
uses an RTX 5080 pilot and a 3×A10 formal study. The pilot has run and
stopped the protocol: at the planned budget the frozen model leaves almost no
room for per-request depth allocation ([pilot report](docs/PILOT_REPORT.v2.1.zh-CN.md)).
The follow-up protocol moves to GSM8K solutions with one depth decision per
solution line: [v3.1 plan](docs/EXPERIMENT_PLAN.v3.1.zh-CN.md). Its frozen-model
phase 0, run online at a matched budget of half the depth-16 cost, found that
per-line or per-question depth allocation beats uniform depth by about 0.03 nat
per token, while the frozen representation adds no detectable gain over a
position-only schedule. Its RTX 5080 pilot stopped the protocol: at equal cost the
jointly trained controller did not beat the best position schedule (-0.0064 nat per
token, 95% interval [-0.0134, 0.0005]); its readout saturated, and its generation
parameters were 0.012 nat per token worse under a fixed schedule
([v3.1 pilot report](docs/PILOT_REPORT.v3.1.zh-CN.md)). The
[v3.2 plan](docs/EXPERIMENT_PLAN.v3.2.zh-CN.md) tested a new readout with the generation
parameters frozen; it stopped at stage 1: +0.0038 nat per token over the position
schedule, 95% interval [-0.0039, 0.0116] ([stage 1 report](docs/PILOT_REPORT.v3.2-stage1.zh-CN.md)).
A matched follow-up continued training from the same adapted parameters with and without the
control loss: at an identical fixed schedule the control loss raised generation NLL by 0.0016
nat per token (95% interval [0.0005, 0.0028]), while the overall policy difference, +0.0038 in
favour of the control arm, did not reach the evidence threshold
([follow-up report](docs/PILOT_REPORT.v3.2-followup.zh-CN.md)). The 3×A10 confirmatory
experiment v3.3 (2,397 documents, strict common cost) found that the model's own representation
beats the position schedule by +0.0019 nat per token (one-sided 95% lower bound +0.0002, upper
end 0.0040): detected, but below the 0.005 practical threshold. Across three seeds the control
gradient raised generation NLL by 0.0017, and separate optimizer states removed about 77% of
that harm ([v3.3 report](docs/EXPERIMENT_REPORT.v3.3.zh-CN.md)).

All experiments from v1 to v4 are consolidated in the
[experiment summary](docs/EXPERIMENT_SUMMARY.zh-CN.md). The v4 row-loop architecture on
Qwen3-30B-A3B ([architecture](docs/ARCHITECTURE.v4.zh-CN.md),
[flow](docs/ARCHITECTURE_FLOW.v4.zh-CN.md), [mathematics](docs/ARCHITECTURE_MATH.v4.zh-CN.md),
[one-day plan](docs/EXPERIMENT_PLAN.v4-one-day.zh-CN.md)) has been reviewed against these
results ([review](docs/ARCHITECTURE_REVIEW.v4.zh-CN.md)) and frozen for implementation as
protocol `jev-v4-rowloop` ([configuration](docs/EXPERIMENT_CONFIG.v4-rowloop.json)). No v4 run
has started.

## 简体中文

核心想法是把路由写成一份可执行的计算计划：**哪些 token 值得继续计算、
由哪些专家计算、追加多少递归深度。** JEV使用本家[Canopy-Jev](https://github.com/Camellia86/Canopy-Jev)实现，
输入完整合法上下文、专家能力卡、当前深度和执行回执，直接得到数值决策。
目前实现了联合动作选择、收益代理预算分配、执行后重路由、完整计划选择四条路线。

自回归模式按位置逐步提交决策；允许全输入的任务才使用全局预算优化。
代码包含真实 Canopy 接口、预算与容量约束、稀疏执行接口、蒸馏损失和可运行例子。
本版本发布的是研究算法实现，还没有真实 MoE/MoR 的训练收益结论。
数学设计由用户指定的 GPT-6-astra、max 推理子代理完成并记录在研究文档中。

另有一份独立的[共享末端深度控制实验计划](docs/EXPERIMENT_PLAN.zh-CN.md)：在冻结的 Huginn-0125 循环模型上训练共享的末端两层 LoRA，
让同一段前向同时完成生成和循环深度选择。它不调用 Canopy-Jev，也不使用本仓库的路由代码；
RTX 5080 先导已经运行并判定不放行：在计划的预算点上，冻结模型几乎没有可分配给不同请求的深度空间，详见[先导报告](docs/PILOT_REPORT.v2.1.zh-CN.md)。后续的 [v3.1 计划](docs/EXPERIMENT_PLAN.v3.1.zh-CN.md)改在 GSM8K 解答上按步骤决定深度；冻结模型的第 0 阶段已按在线语义重做。v3.1 的 RTX 5080 先导判定不放行：同成本下，联合训练的控制器没有超过最好的位置日程（−0.0064 nat/token，95% 区间 [−0.0134, 0.0005]）；它的读出饱和，生成参数在固定日程下差 0.012 nat/token，详见 [v3.1 先导报告](docs/PILOT_REPORT.v3.1.zh-CN.md)。[v3.2 计划](docs/EXPERIMENT_PLAN.v3.2.zh-CN.md)在冻结的生成参数上检验新读出，阶段一不放行：相对位置日程改善 +0.0038 nat/token，95% 区间 [−0.0039, 0.0116]，详见[阶段一报告](docs/PILOT_REPORT.v3.2-stage1.zh-CN.md)。追加的匹配实验从同一组已适配参数出发，只差控制损失：同一固定日程下控制损失使生成 NLL 升高 0.0016 nat/token（95% 区间 [0.0005, 0.0028]）；整体策略差 +0.0038 偏向含控制损失的一臂，未达到证据门槛，详见[追加实验报告](docs/PILOT_REPORT.v3.2-followup.zh-CN.md)。v3.3 在 3×A10 上做了确认性实验（2,397 篇，严格同成本）：模型自身表示相对位置日程改善 +0.0019 nat/token，单侧 95% 下界 +0.0002，区间上端 0.0040，检出增益但低于 0.005 的实用门槛；三个种子上控制梯度使生成 NLL 升高 0.0017，分开维护优化器状态消除了其中约 77%，详见[正式实验报告](docs/EXPERIMENT_REPORT.v3.3.zh-CN.md)。

全部实验（v1 到 v4）的汇总见[实验汇总](docs/EXPERIMENT_SUMMARY.zh-CN.md)。v4 在 Qwen3-30B-A3B 上的按行循环架构（[架构正文](docs/ARCHITECTURE.v4.zh-CN.md)、[流程](docs/ARCHITECTURE_FLOW.v4.zh-CN.md)、[数学验证](docs/ARCHITECTURE_MATH.v4.zh-CN.md)、[一天方案](docs/EXPERIMENT_PLAN.v4-one-day.zh-CN.md)）已按这些结果完成[审核](docs/ARCHITECTURE_REVIEW.v4.zh-CN.md)，定稿为协议 `jev-v4-rowloop`（[配置](docs/EXPERIMENT_CONFIG.v4-rowloop.json)），尚未开始运行。

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
