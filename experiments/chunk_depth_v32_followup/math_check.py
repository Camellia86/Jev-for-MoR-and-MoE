"""CPU / float64 核验 E-C 的逐行完整控制 logit 梯度。

默认只向标准输出打印 JSON；--output 显式指定回执文件。模块不导入 common，
不加载模型、数据或 CUDA，不创建目录。成本和软标签视为停止梯度的常量。
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

K = 5
WEIGHTS = {"route": 1.0, "ce": 0.1, "brier": 0.01, "kl": 0.1}
AUTOGRAD_ATOL, DIRECTION_ATOL = 1e-10, 1e-7
SEED = 20261013


def component_values(logits, cost_per_token, soft_label, reference_logp):
    """直接按训练定义写目标；C、s、reference 均由调用者固定。"""
    logp = torch.log_softmax(logits, -1)
    p = logp.exp()
    return {
        "route": (p * cost_per_token).sum(),
        "ce": -(soft_label * logp).sum(),
        "brier": ((p - soft_label) ** 2).sum(),
        "kl": (p * (logp - reference_logp)).sum(),
    }


def analytic_gradients(logits, cost_per_token, soft_label, reference_logp):
    """独立解析表达式；不调用 autograd。"""
    logp = torch.log_softmax(logits.detach(), -1)
    p = logp.exp()
    error = p - soft_label
    log_ratio = logp - reference_logp
    return {
        "route": p * (cost_per_token - (p * cost_per_token).sum()),
        "ce": p - soft_label,
        "brier": 2 * p * (error - (p * error).sum()),
        "kl": p * (log_ratio - (p * log_ratio).sum()),
    }


def check(seed=SEED, cases=16):
    generator = torch.Generator(device="cpu").manual_seed(seed)
    dtype = torch.float64
    costs = torch.tensor([0.301, 0.418, 0.534, 0.767, 1.0], dtype=dtype, device="cpu")
    records = []
    for index in range(cases):
        logits = (2 * torch.randn(K, generator=generator, dtype=dtype, device="cpu")).requires_grad_(True)
        base = 0.1 + 1.5 * torch.rand(K, generator=generator, dtype=dtype, device="cpu")
        kv = -0.04 + 0.18 * torch.rand(K, generator=generator, dtype=dtype, device="cpu")
        lam, tau = 1.0, 0.05
        cost = (base + lam * costs + kv).detach()
        s = torch.softmax(-cost / tau, -1).detach()
        reference_logp = torch.log_softmax(
            torch.randn(K, generator=generator, dtype=dtype, device="cpu"), -1).detach()
        n, N = (3, 11, 37, 71)[index % 4], 113
        w = n / N
        analytic = analytic_gradients(logits, cost, s, reference_logp)
        values = component_values(logits, cost, s, reference_logp)
        auto_error = {}
        for name in WEIGHTS:
            observed = torch.autograd.grad(values[name], logits, retain_graph=True)[0]
            auto_error[name] = float((observed - analytic[name]).abs().max())
        # 与训练等价：route 中的 C_total = n * c，每个辅助项乘 n，再统一除 N。
        full = ((logits.softmax(-1) * (n * cost)).sum()
                + n * sum(WEIGHTS[k] * values[k] for k in ("ce", "brier", "kl"))) / N
        expected = w * sum(WEIGHTS[k] * analytic[k] for k in WEIGHTS)
        observed = torch.autograd.grad(full, logits, retain_graph=True)[0]
        auto_error["weighted_full"] = float((observed - expected).abs().max())
        direction = torch.randn(K, generator=generator, dtype=dtype, device="cpu")
        direction /= direction.norm()
        step = 1e-5
        plus = component_values(logits.detach() + step * direction, cost, s, reference_logp)
        minus = component_values(logits.detach() - step * direction, cost, s, reference_logp)
        finite_error = {}
        for name in WEIGHTS:
            finite = (plus[name] - minus[name]) / (2 * step)
            exact = (analytic[name] * direction).sum()
            finite_error[name] = float((finite - exact).abs())
        full_finite = w * sum(WEIGHTS[k] * (plus[k] - minus[k]) for k in WEIGHTS) / (2 * step)
        finite_error["weighted_full"] = float((full_finite - (expected * direction).sum()).abs())
        shift = 7.25
        shifted_loss = (logits.softmax(-1) * (cost + shift)).sum()
        shifted = torch.autograd.grad(shifted_loss, logits)[0]
        shift_error = float((shifted - analytic["route"]).abs().max())
        zero_sum_error = max(float(g.sum().abs()) for g in analytic.values())
        p, ref = logits.detach().softmax(-1), reference_logp.exp()
        assert float(p.max() - p.min()) > 1e-3
        assert float(s.max() - s.min()) > 1e-3
        assert float(ref.max() - ref.min()) > 1e-3
        assert all(math.isfinite(e) and e <= AUTOGRAD_ATOL for e in auto_error.values()), auto_error
        assert all(math.isfinite(e) and e <= DIRECTION_ATOL for e in finite_error.values()), finite_error
        assert shift_error <= AUTOGRAD_ATOL, shift_error
        assert zero_sum_error <= AUTOGRAD_ATOL, zero_sum_error
        records.append({"case": index, "line_tokens": n, "document_tokens": N,
                        "p": p.tolist(), "s": s.tolist(), "reference": ref.tolist(),
                        "base_cost": base.tolist(), "compute_cost": (lam * costs).tolist(),
                        "kv_cost": kv.tolist(), "autograd_max_abs_error": auto_error,
                        "central_difference_abs_error": finite_error,
                        "route_constant_shift_max_abs_error": shift_error,
                        "gradient_zero_sum_max_abs_error": zero_sum_error})
    return {"kind": "chunk_depth_v32_followup_control_math", "status": "passed", "device": "cpu",
            "dtype": "float64", "torch_version": torch.__version__, "seed": seed, "cases": cases,
            "autograd_tolerance": AUTOGRAD_ATOL, "central_difference_tolerance": DIRECTION_ATOL,
            "central_difference_step": 1e-5, "control_weights": WEIGHTS,
            "max_autograd_error": max(max(r["autograd_max_abs_error"].values()) for r in records),
            "max_directional_error": max(max(r["central_difference_abs_error"].values()) for r in records),
            "max_route_constant_shift_error": max(r["route_constant_shift_max_abs_error"] for r in records),
            "note": "给定冻结成本、软标签、参考和采样历史的局部控制目标；不声称序列回报无偏性。",
            "records": records}


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="可选 JSON 回执路径；其父目录须已存在")
    parser.add_argument("--no-write", action="store_true", help="明确禁止写文件，仅打印 JSON")
    args = parser.parse_args()
    if args.no_write and args.output is not None:
        parser.error("--no-write 与 --output 不能同时使用")
    result = check()
    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output is not None:
        # 独占创建，避免覆盖已完成的数学回执。
        with args.output.open("x", encoding="utf-8") as stream:
            stream.write(encoded + "\n")
    print(encoded)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
