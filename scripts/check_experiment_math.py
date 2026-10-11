"""CPU-only algebra checks for the proposed shared-coda depth-control protocol.

This script does not load Huginn, train an adapter, execute a GPU workload, or
measure model quality or hardware capacity. Its JSON records only the checks
actually performed below. Run with the project's Python environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch


SEED = 20261011
ACTION_COUNT = 4
KAPPA = 256.0
EPSILON = 0.01
ETA = 0.001
SAMPLES_PER_STATE = 64
MONTE_CARLO_REPEATS = 4096
DTYPE = torch.float64


def vector(value: torch.Tensor) -> list[float]:
    return value.detach().cpu().tolist()


def distribution(logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    p = logits.softmax(-1)
    alpha = KAPPA * ((1 - EPSILON) * p + EPSILON / p.numel()) + ETA
    mu = alpha / alpha.sum()
    return p, alpha, mu


def check_close(actual: torch.Tensor, expected: torch.Tensor, *, atol: float = 1e-10) -> float:
    error = float((actual - expected).abs().max().detach())
    if not torch.allclose(actual, expected, rtol=1e-8, atol=atol):
        raise AssertionError(f"Maximum absolute error {error} exceeds tolerance")
    return error


def check_dirichlet() -> dict:
    logits = torch.tensor([0.15, 0.25, 0.35, 0.25], dtype=DTYPE).log().requires_grad_()
    cost = torch.tensor([0.2, 0.0, 0.7, 0.35], dtype=DTYPE)
    p, alpha, mu = distribution(logits)
    total = KAPPA + ACTION_COUNT * ETA
    beta = KAPPA * (1 - EPSILON) / total
    identity = beta * p + (1 - beta) / ACTION_COUNT
    identity_error = check_close(mu, identity)
    objective = (mu * cost).sum()
    autograd_gradient = torch.autograd.grad(objective, logits)[0]
    analytic_gradient = beta * p.detach() * (cost - (p.detach() * cost).sum())
    analytic_error = check_close(autograd_gradient, analytic_gradient)
    step = 1e-5
    finite_difference = torch.empty_like(logits)
    for index in range(ACTION_COUNT):
        offset = torch.zeros_like(logits)
        offset[index] = step
        plus = (distribution(logits.detach() + offset)[2] * cost).sum()
        minus = (distribution(logits.detach() - offset)[2] * cost).sum()
        finite_difference[index] = (plus - minus) / (2 * step)
    finite_difference_error = check_close(finite_difference, analytic_gradient, atol=1e-9)

    samples = torch.distributions.Dirichlet(alpha.detach()).sample(
        (MONTE_CARLO_REPEATS, SAMPLES_PER_STATE)
    )
    alpha_check = distribution(logits)[1]
    log_density = torch.distributions.Dirichlet(alpha_check).log_prob(samples[0, 0])
    autograd_score = torch.autograd.grad(log_density, logits)[0]
    u = samples.log() - torch.digamma(alpha.detach())
    score = KAPPA * (1 - EPSILON) * p.detach() * (
        u - (u * p.detach()).sum(-1, keepdim=True)
    )
    score_error = check_close(score[0, 0], autograd_score, atol=1e-9)

    def loo_gradient(candidate_cost: torch.Tensor) -> torch.Tensor:
        sample_cost = samples @ candidate_cost
        advantage = SAMPLES_PER_STATE / (SAMPLES_PER_STATE - 1) * (
            sample_cost - sample_cost.mean(1, keepdim=True)
        )
        return (advantage.unsqueeze(-1) * score).mean(1)

    gradients = loo_gradient(cost)
    mean_gradient = gradients.mean(0)
    standard_error = gradients.std(0, unbiased=True) / math.sqrt(MONTE_CARLO_REPEATS)
    z_error = (mean_gradient - analytic_gradient) / standard_error
    if not bool((z_error.abs() < 5.0).all()):
        raise AssertionError("LOO mean differs from the exact gradient by >= 5 standard errors")
    mean_q = samples.mean((0, 1))
    mean_q_se = samples.reshape(-1, ACTION_COUNT).std(0, unbiased=True) / math.sqrt(
        MONTE_CARLO_REPEATS * SAMPLES_PER_STATE
    )
    mean_q_z = (mean_q - mu.detach()) / mean_q_se
    if not bool((mean_q_z.abs() < 5.0).all()):
        raise AssertionError("Empirical Dirichlet mean differs by >= 5 standard errors")

    constant_gradient = loo_gradient(torch.full_like(cost, 3.2))
    constant_max_error = float(constant_gradient.abs().max())
    if constant_max_error > 1e-10:
        raise AssertionError("Constant costs must have zero policy gradient")
    translated_gradient = loo_gradient(cost + 17.75)
    translation_error = check_close(translated_gradient, gradients, atol=1e-10)
    analytic_constant = beta * p.detach() * (3.2 - torch.tensor(3.2, dtype=DTYPE))
    check_close(analytic_constant, torch.zeros_like(cost))

    return {
        "status": "passed",
        "p": vector(p), "cost": vector(cost), "alpha": vector(alpha),
        "total_concentration": total, "beta": beta,
        "mean_identity_max_abs_error": identity_error,
        "autograd_vs_closed_form_max_abs_error": analytic_error,
        "finite_difference_step": step,
        "finite_difference_max_abs_error": finite_difference_error,
        "score_vs_log_density_autograd_max_abs_error": score_error,
        "exact_logit_gradient": vector(analytic_gradient),
        "loo_mean_logit_gradient": vector(mean_gradient),
        "loo_mean_standard_error": vector(standard_error),
        "loo_mean_error_in_standard_errors": vector(z_error),
        "monte_carlo_acceptance_max_abs_standard_errors": 5.0,
        "loo_gradient_cosine": float(torch.nn.functional.cosine_similarity(
            mean_gradient, analytic_gradient, dim=0)),
        "loo_gradient_relative_error": float(torch.linalg.vector_norm(
            mean_gradient - analytic_gradient) / torch.linalg.vector_norm(analytic_gradient)),
        "empirical_q_mean": vector(mean_q),
        "empirical_q_mean_error_in_standard_errors": vector(mean_q_z),
        "constant_cost_max_abs_sample_gradient": constant_max_error,
        "cost_translation": 17.75,
        "translation_max_abs_gradient_error": translation_error,
        "interpretation": "The LOO estimator is checked against the same exact fixed-cost objective; this is not evidence of a quality advantage from sampling.",
    }


def check_joint_gradient() -> dict:
    scalar = torch.tensor(0.0, dtype=DTYPE, requires_grad=True)
    probability = scalar.sigmoid()
    binary_mu = torch.stack((probability, 1 - probability))
    binary_cost = torch.stack((scalar.square(), (scalar - 1).square()))
    objective = (binary_mu * binary_cost).sum()
    full = torch.autograd.grad(objective, scalar, retain_graph=True)[0]
    direct = torch.autograd.grad((binary_mu.detach() * binary_cost).sum(), scalar, retain_graph=True)[0]
    route = torch.autograd.grad((binary_mu * binary_cost.detach()).sum(), scalar)[0]
    check_close(full, torch.tensor(-1.25, dtype=DTYPE))
    check_close(direct, torch.tensor(-1.0, dtype=DTYPE))
    check_close(route, torch.tensor(-0.25, dtype=DTYPE))
    check_close(full, direct + route)

    theta = torch.tensor(0.17, dtype=DTYPE, requires_grad=True)
    slope = torch.tensor([0.7, -0.4, 0.2, 0.5], dtype=DTYPE)
    offset = torch.tensor([-0.2, 0.3, 0.1, -0.1], dtype=DTYPE)
    centers = torch.tensor([-0.3, 0.2, 0.7, 1.0], dtype=DTYPE)
    extra = torch.tensor([0.1, 0.2, 0.0, 0.05], dtype=DTYPE)
    mu = distribution(slope * theta + offset)[2]
    costs = (theta - centers).square() + extra
    full_objective = (mu * costs).sum()
    direct_surrogate = (mu.detach() * costs).sum()
    routing_surrogate = (mu * costs.detach()).sum()
    full_gradient = torch.autograd.grad(full_objective, theta, retain_graph=True)[0]
    direct_gradient = torch.autograd.grad(direct_surrogate, theta, retain_graph=True)[0]
    routing_gradient = torch.autograd.grad(routing_surrogate, theta, retain_graph=True)[0]
    split_gradient = torch.autograd.grad(direct_surrogate + routing_surrogate, theta)[0]
    error = check_close(full_gradient, split_gradient)
    return {
        "status": "passed",
        "binary_example_at_theta_zero": {
            "full_gradient": float(full), "direct_gradient": float(direct),
            "routing_gradient": float(route),
        },
        "four_action_smoothed_example": {
            "theta": float(theta.detach()), "objective": float(full_objective.detach()),
            "full_gradient": float(full_gradient), "direct_gradient": float(direct_gradient),
            "routing_gradient": float(routing_gradient), "split_gradient": float(split_gradient),
            "max_abs_error": error,
        },
        "surrogate_value_warning": "The direct and routing surrogate scalar values add to twice the objective; only their stopped-gradient derivatives are combined. Log the actual sum(mu * C) separately.",
        "offline_cost_condition": "C must be recomputed for changed shared generation parameters; frozen core latents may be reused, frozen old costs may not.",
    }


def check_budget_derivative() -> dict:
    mu = torch.tensor([0.15, 0.25, 0.35, 0.25], dtype=DTYPE)
    work = torch.tensor([0.35, 0.6, 1.0, 1.8], dtype=DTYPE)
    offset = torch.tensor(0.7, dtype=DTYPE, requires_grad=True)
    calibrated = torch.softmax(mu.log() - offset * work, dim=0)
    expected_work = (calibrated * work).sum()
    derivative = torch.autograd.grad(expected_work, offset)[0]
    variance = (calibrated * (work - expected_work.detach()).square()).sum()
    error = check_close(derivative, -variance)
    if not float(derivative) < 0:
        raise AssertionError("Nonconstant costs must give a negative soft-budget derivative")
    return {
        "status": "passed", "offset": float(offset.detach()),
        "calibrated_probabilities": vector(calibrated),
        "expected_work": float(expected_work.detach()),
        "autograd_budget_derivative": float(derivative),
        "negative_cost_variance": float(-variance.detach()),
        "max_abs_error": error,
        "scope": "This derivative is for categorical probabilities mu_b proportional to mu * exp(-b*f). Hard argmax work is a monotone step function, not a differentiable variance identity.",
    }


def check_lora_initialization() -> dict:
    generator = torch.Generator(device="cpu").manual_seed(SEED + 1)
    batch, n_in, n_out, rank = 5, 8, 6, 3
    x = torch.randn(batch, n_in, generator=generator, dtype=DTYPE)
    base = torch.randn(n_out, n_in, generator=generator, dtype=DTYPE)
    target = torch.randn(batch, n_out, generator=generator, dtype=DTYPE)

    def gradients(both_zero: bool) -> tuple[float, float, float]:
        a_initial = torch.zeros(rank, n_in, dtype=DTYPE) if both_zero else (
            0.01 * torch.randn(rank, n_in, generator=generator, dtype=DTYPE)
        )
        a = a_initial.requires_grad_()
        b = torch.zeros(n_out, rank, dtype=DTYPE, requires_grad=True)
        residual = (b @ a) / rank
        prediction = x @ (base + residual).T
        loss = (prediction - target).square().mean()
        ga, gb = torch.autograd.grad(loss, (a, b))
        return float(ga.norm()), float(gb.norm()), float(residual.detach().abs().max())

    a_norm, b_norm, residual = gradients(False)
    both_a, both_b, both_residual = gradients(True)
    if a_norm != 0.0 or b_norm <= 0.0 or residual != 0.0:
        raise AssertionError("Random A / zero B must preserve the base and permit a first-step B update")
    if both_a != 0.0 or both_b != 0.0 or both_residual != 0.0:
        raise AssertionError("Both-zero LoRA factors must have zero first-step gradients")
    return {
        "status": "passed", "dimensions": {"input": n_in, "output": n_out, "rank": rank},
        "random_a_zero_b": {"a_gradient_norm": a_norm, "b_gradient_norm": b_norm,
                            "initial_residual_max_abs": residual},
        "both_zero": {"a_gradient_norm": both_a, "b_gradient_norm": both_b,
                      "initial_residual_max_abs": both_residual},
    }


def check_resource_arithmetic() -> dict:
    hidden, intermediate, coda_layers, rank = 5280, 17920, 2, 8
    projections = {
        "fused_qkv": [hidden, 3 * hidden],
        "attention_output": [hidden, hidden],
        "fused_gate_up": [hidden, 2 * intermediate],
        "mlp_down": [intermediate, hidden],
    }
    parameters_per_layer = sum(rank * sum(shape) for shape in projections.values())
    parameters = coda_layers * parameters_per_layer
    assert parameters == 1_536_000
    # These alternative targets do NOT implement the fused-projection design.
    separate_q_o_gate_up_down = coda_layers * rank * (7 * hidden + 3 * intermediate)
    assert separate_q_o_gate_up_down == 1_451_520
    prefix, horizon, document_count, bytes_per_element = 128, 128, 5664, 2
    positions = prefix + horizon - 1
    depths = [4, 8, 16, 32]
    latent_bytes_per_document = (len(depths) * positions + prefix) * hidden * bytes_per_element
    assert positions == 255
    assert latent_bytes_per_document == 12_122_880
    latent_bytes_total = latent_bytes_per_document * document_count
    resource_rows = []
    for depth in depths:
        logical_layers = 2 + 4 * depth + 2
        kv_bytes = 2 * logical_layers * positions * hidden * bytes_per_element
        resource_rows.append({
            "depth": depth, "candidate_core_token_steps": depth * positions,
            "logical_attention_layers": logical_layers, "dense_kv_bytes": kv_bytes,
            "dense_kv_mib": kv_bytes / 2**20,
        })
    return {
        "status": "passed", "hidden_size": hidden, "intermediate_size": intermediate,
        "coda_layers": coda_layers, "lora_rank": rank,
        "projection_shapes_in_out": projections,
        "lora_parameters_per_layer": parameters_per_layer,
        "lora_parameters_total": parameters,
        "alternative_separate_q_o_gate_up_down_parameters": separate_q_o_gate_up_down,
        "prefix_tokens": prefix, "target_tokens": horizon,
        "consumed_input_positions": positions,
        "candidate_core_token_step_formula": "depth * (prefix + target - 1)",
        "latent_storage_formula_bytes": "(4 * 255 + 128) * 5280 * 2",
        "latent_storage_per_document_bytes": latent_bytes_per_document,
        "documents": document_count, "latent_storage_total_bytes": latent_bytes_total,
        "latent_storage_total_gib": latent_bytes_total / 2**30,
        "dense_kv_formula_bytes": "2 * (2 + 4 * depth + 2) * 255 * 5280 * 2",
        "per_depth": resource_rows,
        "scope": "Arithmetic for specified shapes only. KV assumes distinct full-width K/V for all logical layers, with no allocator/workspace overhead. Candidate step counts exclude any additional controller-prefix execution. These are not measured VRAM or throughput results.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                        / "docs" / "EXPERIMENT_MATH_CHECK.v1.json")
    args = parser.parse_args()
    torch.set_default_dtype(DTYPE)
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    result = {
        "protocol_id": "jev-shared-depth-v1",
        "kind": "cpu_mathematical_self_check_not_model_experiment",
        "status": "running",
        "run_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "runtime": {"python": sys.version.split()[0], "torch": torch.__version__,
                    "platform": platform.system(), "device": "cpu", "dtype": "float64",
                    "threads": 1, "deterministic_algorithms": True},
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "constants": {"action_count": ACTION_COUNT, "kappa": KAPPA,
                      "epsilon": EPSILON, "eta": ETA,
                      "samples_per_state": SAMPLES_PER_STATE,
                      "monte_carlo_repeats": MONTE_CARLO_REPEATS},
        "checks": {},
        "model_loaded": False, "model_training_performed": False,
        "gpu_benchmark_performed": False,
    }
    checks = {
        "dirichlet_mean_and_score_gradient": check_dirichlet,
        "joint_generation_and_routing_gradient": check_joint_gradient,
        "soft_budget_calibration_derivative": check_budget_derivative,
        "lora_zero_residual_initialization": check_lora_initialization,
        "parameter_and_storage_arithmetic": check_resource_arithmetic,
    }
    for name, check in checks.items():
        try:
            result["checks"][name] = check()
        except Exception as exc:
            result["checks"][name] = {"status": "failed", "error_type": type(exc).__name__,
                                      "error": str(exc)}
    result["status"] = "passed" if all(
        item["status"] == "passed" for item in result["checks"].values()
    ) else "failed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": result["status"], "checks": {
        name: item["status"] for name, item in result["checks"].items()},
        "output": str(args.output.resolve())}, ensure_ascii=False))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
