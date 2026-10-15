"""Deterministic, standard-library checks of the v4 architecture equations.

This is a four-dimensional algebraic model, not a Qwen implementation.  The
online implementation stores K/V.  The independent reference stores layer
inputs, projects the complete prefix afresh, and explicitly applies a causal
mask.  Both implement the *specified* last-actual-round history completion;
neither claims that this is the same as executing all unselected deep rounds.

Run with Python 3.10+; no model weights or external packages are needed.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random


SEED = 40723
WIDTH = 4
F_LAYERS = 2
TAIL_LAYERS = 2
TOL = 1e-10
NEWLINE = 0


def dot(a, b):
    return sum(x * y for x, y in zip(a, b))


def mv(matrix, vector):
    return [dot(row, vector) for row in matrix]


def add(a, b):
    return [x + y for x, y in zip(a, b)]


def rope(vector, position):
    """RoPE position is the absolute input position, independent of round."""
    result = []
    for pair, index in enumerate(range(0, WIDTH, 2)):
        angle = position / (10.0 ** pair)
        c, s = math.cos(angle), math.sin(angle)
        x, y = vector[index:index + 2]
        result.extend([c * x - s * y, s * x + c * y])
    return result


def softmax(values):
    high = max(values)
    exponentials = [math.exp(value - high) for value in values]
    total = sum(exponentials)
    return [value / total for value in exponentials]


def max_error(a, b):
    if isinstance(a, dict):
        return max((max_error(a[key], b[key]) for key in a), default=0.0)
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return float("inf")
        return max((max_error(x, y) for x, y in zip(a, b)), default=0.0)
    return abs(a - b)


def embed(token):
    return [0.4 * math.sin((token + 1) * (j + 1) * 0.37)
            + 0.13 * math.cos((token + 2) * (j + 1)) for j in range(WIDTH)]


def matrix(rng, scale):
    return [[rng.uniform(-scale, scale) for _ in range(WIDTH)]
            for _ in range(WIDTH)]


def layer_weights(rng):
    return {name: matrix(rng, 0.3 if name != "ff" else 0.15)
            for name in ("q", "k", "v", "o", "ff")}


def adapt(weights, rng, scale):
    """Rank-one LoRA B A is applied to each matrix, including K and V."""
    result = copy.deepcopy(weights)
    for name in weights:
        a = [rng.uniform(-0.5, 0.5) for _ in range(WIDTH)]
        b = [rng.uniform(-0.5, 0.5) for _ in range(WIDTH)]
        for i in range(WIDTH):
            for j in range(WIDTH):
                result[name][i][j] += scale * b[i] * a[j]
    return result


class Model:
    def __init__(self, zero=False):
        rng = random.Random(SEED)
        self.shared = layer_weights(rng)
        self.base_loop = [layer_weights(rng) for _ in range(F_LAYERS)]
        self.repeat_loop = [adapt(w, rng, 0.0 if zero else 0.7)
                            for w in self.base_loop]
        self.gamma = [0.0] * WIDTH if zero else [0.2, 0.4, -0.1, 0.65]
        base_tail = [layer_weights(rng) for _ in range(TAIL_LAYERS)]
        self.tails = [[adapt(w, rng, 0.0 if zero else 0.35 * expert)
                       for w in base_tail] for expert in range(3)]
        self.decision = layer_weights(rng)
        self.action_head = [
            [rng.uniform(-0.1, 0.1) for _ in range(WIDTH)] for _ in range(12)]
        self.lm = [
            [rng.uniform(-0.6, 0.6) for _ in range(WIDTH)] for _ in range(7)]

    def loop_weights(self, round_index):
        return self.base_loop if round_index == 0 else self.repeat_loop

    def mixed_input(self, prefix, previous):
        return [u + g * (h - u)
                for u, g, h in zip(prefix, self.gamma, previous)]

    def action(self, decision_state, boundary_index, schedule):
        # Fixed causal boundary-index biases exercise all depths and tails.
        # D's nonzero trainable linear readout is included.  This is a test
        # fixture, not a claim about a learned controller's policy quality.
        intended = schedule[boundary_index % len(schedule)]
        preferred = (intended[0] - 1) * 3 + intended[1]
        scores = mv(self.action_head, decision_state)
        scores[preferred] += 2.0
        winner = max(range(len(scores)), key=scores.__getitem__)
        return (winner // 3 + 1, winner % 3)


def online_layer(weights, x, position, cache):
    """One-token attention with an explicit immutable position in each slot."""
    assert len(cache) == position
    q = rope(mv(weights["q"], x), position)
    k = rope(mv(weights["k"], x), position)
    v = mv(weights["v"], x)
    cache.append((position, k, v))
    assert [entry[0] for entry in cache] == list(range(position + 1))
    scores = [dot(q, key) / math.sqrt(WIDTH) for _, key, _ in cache]
    probabilities = softmax(scores)
    context = [sum(p * entry[2][j] for p, entry in zip(probabilities, cache))
               for j in range(WIDTH)]
    residual = add(x, mv(weights["o"], context))
    return add(residual, [0.2 * math.tanh(v)
                          for v in mv(weights["ff"], residual)])


def batch_layer(weights, inputs):
    """Independent full matrix attention reference with an explicit mask."""
    queries = [rope(mv(weights["q"], x), t) for t, x in enumerate(inputs)]
    keys = [rope(mv(weights["k"], x), t) for t, x in enumerate(inputs)]
    values = [mv(weights["v"], x) for x in inputs]
    result = []
    for t in range(len(inputs)):
        # Compute all scores, then select the causal lower-triangular part.
        scores = [sum(queries[t][j] * keys[s][j] for j in range(WIDTH))
                  / math.sqrt(WIDTH) for s in range(len(inputs))]
        weights_t = softmax(scores[:t + 1])
        attended = [sum(weights_t[s] * values[s][j] for s in range(t + 1))
                    for j in range(WIDTH)]
        residual = [inputs[t][j] + sum(weights["o"][j][k] * attended[k]
                                      for k in range(WIDTH))
                    for j in range(WIDTH)]
        result.append([residual[j] + 0.2 * math.tanh(sum(
            weights["ff"][j][k] * residual[k] for k in range(WIDTH)))
            for j in range(WIDTH)])
    return result


def empty_cost():
    return dict(shared_layer_tokens=0, actual_loop_layer_tokens=0,
                missing_loop_kv_slots=0, missing_loop_projection_matvecs=0,
                decision_layer_tokens=0, tail_current_layer_tokens=0,
                tail_catchup_layer_tokens=0, lm_readouts=0)


def execute_online(model, tokens, prompt_end, schedule, wrong_backfill=False):
    shared_cache, decision_cache = [], []
    loop_cache = [[[] for _ in range(F_LAYERS)] for _ in range(4)]
    tail_cache = [[[] for _ in range(TAIL_LAYERS)] for _ in model.tails]
    records, final_exits, first_exits = [], [], []
    logits, actions, decisions, events, decision_states = [], [], [], [], []
    cost = empty_cost()
    active = (1, 0)
    boundary_index = 0
    for t, token in enumerate(tokens):
        prefix = online_layer(model.shared, embed(token), t, shared_cache)
        cost["shared_layer_tokens"] += 1
        round_inputs = []

        def run_round(k, x):
            layer_inputs = []
            for layer, weights in enumerate(model.loop_weights(k)):
                cache = loop_cache[k][layer]
                for s in range(len(cache), t):
                    source_round = min(k, len(records[s]) - 1)
                    source = (final_exits[s] if wrong_backfill
                              else records[s][source_round][layer])
                    cache.append((s, rope(mv(weights["k"], source), s),
                                  mv(weights["v"], source)))
                    cost["missing_loop_kv_slots"] += 1
                    cost["missing_loop_projection_matvecs"] += 2
                layer_inputs.append(x)
                x = online_layer(weights, x, t, cache)
                cost["actual_loop_layer_tokens"] += 1
            round_inputs.append(layer_inputs)
            return x

        h = run_round(0, prefix)
        first_exits.append(h)
        state = online_layer(model.decision, h, t, decision_cache)
        decision_states.append(state)
        cost["decision_layer_tokens"] += 1
        boundary = t == prompt_end or (t > prompt_end and token == NEWLINE)
        if boundary:
            active = model.action(state, boundary_index, schedule)
            boundary_index += 1
            decisions.append((t, *active))
            events.append((t, "decision", *active))
        actions.append(active)
        depth, expert = active
        for k in range(1, depth):
            h = run_round(k, model.mixed_input(prefix, h))
        records.append(round_inputs)
        final_exits.append(h)
        # Prompt prefill saves first-round exits.  No generation tail or
        # vocabulary head runs before the prompt's final input position.
        if t < prompt_end:
            continue
        # The selected tail catches up from its own cursor using the saved
        # final loop exits, never another tail's K/V or hidden states.
        cursor = len(tail_cache[expert][0])
        for s in range(cursor, t + 1):
            tail_h = final_exits[s]
            for layer, weights in enumerate(model.tails[expert]):
                tail_h = online_layer(weights, tail_h, s, tail_cache[expert][layer])
                cost["tail_current_layer_tokens" if s == t
                     else "tail_catchup_layer_tokens"] += 1
        logits.append(mv(model.lm, tail_h))
        cost["lm_readouts"] += 1
        events.append((t, "logits", *active))
    return dict(logits=logits, actions=actions, decisions=decisions,
                logit_input_positions=list(range(prompt_end, len(tokens))),
                first_exits=first_exits, decision_states=decision_states,
                cost=cost, events=events)


def execute_reference(model, tokens, prompt_end, schedule):
    shared = batch_layer(model.shared, [embed(token) for token in tokens])
    records, final_exits, first_exits = [], [], []
    logits, actions, decisions, decision_states = [], [], [], []
    active = (1, 0)
    boundary_index = 0
    for t, token in enumerate(tokens):
        round_inputs = []

        def run_round(k, x):
            layer_inputs = []
            for layer, weights in enumerate(model.loop_weights(k)):
                # Reconstruct the full input matrix.  There is no K/V cache.
                history = [past[min(k, len(past) - 1)][layer] for past in records]
                layer_inputs.append(x)
                x = batch_layer(weights, history + [x])[-1]
            round_inputs.append(layer_inputs)
            return x

        h = run_round(0, shared[t])
        first_exits.append(h)
        state = batch_layer(model.decision, first_exits)[-1]
        decision_states.append(state)
        if t == prompt_end or (t > prompt_end and token == NEWLINE):
            active = model.action(state, boundary_index, schedule)
            boundary_index += 1
            decisions.append((t, *active))
        actions.append(active)
        for k in range(1, active[0]):
            h = run_round(k, model.mixed_input(shared[t], h))
        records.append(round_inputs)
        final_exits.append(h)
        if t < prompt_end:
            continue
        sequence = final_exits
        for weights in model.tails[active[1]]:
            sequence = batch_layer(weights, sequence)
        logits.append(mv(model.lm, sequence[-1]))
    return dict(logits=logits, actions=actions, decisions=decisions,
                logit_input_positions=list(range(prompt_end, len(tokens))),
                first_exits=first_exits, decision_states=decision_states)


def baseline(model, tokens):
    sequence = batch_layer(model.shared, [embed(token) for token in tokens])
    for weights in model.base_loop + model.tails[0]:
        sequence = batch_layer(weights, sequence)
    return [mv(model.lm, vector) for vector in sequence]


def gradient_check(at_initialization=False):
    """Shared rank-one LoRA and elementwise mixing over three reentries."""
    u = [0.4, -0.2, 0.7]
    m = [[0.7, 0.1, -0.2], [-0.3, 0.8, 0.1], [0.2, -0.1, 0.6]]
    gamma, a, b = [0.2, 0.5, -0.3], [0.15, -0.25, 0.4], [0.3, 0.1, -0.2]
    if at_initialization:
        gamma, b = [0.0] * 3, [0.0] * 3
    target = [-0.1, 0.3, 0.2]

    def forward(gamma, a, b):
        h = mv(m, u)
        tape = []
        for _ in range(3):
            previous = h
            z = [u[i] + gamma[i] * (previous[i] - u[i]) for i in range(3)]
            h = add(mv(m, z), [bi * dot(a, z) for bi in b])
            tape.append((previous, z))
        loss = 0.5 * sum((x - y) ** 2 for x, y in zip(h, target))
        return loss, h, tape

    _, output, tape = forward(gamma, a, b)
    dh = [x - y for x, y in zip(output, target)]
    dg, da, db = [0.0] * 3, [0.0] * 3, [0.0] * 3
    for previous, z in reversed(tape):
        for j in range(3):
            da[j] += dot(dh, b) * z[j]
            db[j] += dh[j] * dot(a, z)
        dz = [sum(m[i][j] * dh[i] for i in range(3)) + a[j] * dot(b, dh)
              for j in range(3)]
        for j in range(3):
            dg[j] += dz[j] * (previous[j] - u[j])
        dh = [dz[j] * gamma[j] for j in range(3)]
    analytic = dg + da + db
    parameters = gamma + a + b
    numerical = []
    eps = 1e-6
    for j in range(len(parameters)):
        plus, minus = parameters.copy(), parameters.copy()
        plus[j] += eps
        minus[j] -= eps
        lp = forward(plus[:3], plus[3:6], plus[6:])[0]
        lm = forward(minus[:3], minus[3:6], minus[6:])[0]
        numerical.append((lp - lm) / (2.0 * eps))
    return dict(max_absolute_error=max_error(analytic, numerical),
                max_relative_error=max(abs(x - y) / max(1e-10, abs(x), abs(y))
                                       for x, y in zip(analytic, numerical)),
                epsilon=eps, shared_reentries=3,
                analytic=analytic, finite_difference=numerical)


def run_checks():
    tokens = [2, 5, 8, 4, 6, 0, 7, 3, 0, 9, 1, 0, 4, 8]
    prompt_end = 2
    schedule = [(1, 0), (4, 1), (2, 2), (3, 0)]
    model, zero = Model(), Model(zero=True)
    checks = []

    def record(name, error, tolerance=TOL, **details):
        checks.append(dict(name=name, max_error=error, tolerance=tolerance,
                           passed=error <= tolerance, **details))

    original = baseline(zero, tokens)[prompt_end:]
    collapse_error = 0.0
    # Every depth and expert appears in each boundary position across rotations.
    all_actions = [(r, e) for r in range(1, 5) for e in range(3)]
    for shift in range(len(all_actions)):
        rotated = all_actions[shift:] + all_actions[:shift]
        actual = execute_online(zero, tokens, prompt_end, rotated)
        collapse_error = max(collapse_error, max_error(actual["logits"], original))
    record("zero_mix_zero_lora_path_collapse", collapse_error,
           paths=12, tested_depths=[1, 2, 3, 4], tested_experts=[0, 1, 2],
           note="All tail weights equal the original tail in this initialization test.")

    online = execute_online(model, tokens, prompt_end, schedule)
    reference = execute_reference(model, tokens, prompt_end, schedule)
    record("nonzero_online_matches_independent_masked_reference",
           max_error({k: online[k] for k in reference}, reference))

    causal_error = 0.0
    for cut in (2, 5, 8, 11):
        changed = tokens[:cut + 1] + [token + 13 for token in tokens[cut + 1:]]
        other = execute_online(model, changed, prompt_end, schedule)
        prefix_logits = cut - prompt_end + 1
        causal_error = max(causal_error,
                           max_error(online["logits"][:prefix_logits],
                                     other["logits"][:prefix_logits]),
                           max_error(online["decision_states"][:cut + 1],
                                     other["decision_states"][:cut + 1]),
                           max_error([d for d in online["decisions"] if d[0] <= cut],
                                     [d for d in other["decisions"] if d[0] <= cut]))
    record("future_token_invariance", causal_error, cuts=[2, 5, 8, 11])

    expected_boundaries = [2, 5, 8, 11]
    observed_boundaries = [entry[0] for entry in online["decisions"]]
    timing_errors = int(observed_boundaries != expected_boundaries)
    timing_errors += int(any(event[0] < prompt_end for event in online["events"]))
    timing_errors += int(online["logit_input_positions"] != list(range(prompt_end, len(tokens))))
    for boundary in expected_boundaries:
        events = [event[1] for event in online["events"] if event[0] == boundary]
        timing_errors += int(events != ["decision", "logits"])
    for begin, end in zip(expected_boundaries, expected_boundaries[1:] + [len(tokens)]):
        timing_errors += int(any(action != online["actions"][begin]
                                 for action in online["actions"][begin:end]))
    record("prompt_and_newline_boundary_order_and_line_constant_action",
           timing_errors, tolerance=0.0, input_boundary_positions=expected_boundaries,
           predicted_first_token_positions=[t + 1 for t in expected_boundaries])

    changed_depth = execute_online(model, tokens, prompt_end, [(4, 0)])
    changed_tail = execute_online(model, tokens, prompt_end, [(1, 2)])
    depth_effect = max_error(online["logits"][0], changed_depth["logits"][0])
    tail_effect = max_error(online["logits"][0], changed_tail["logits"][0])
    record("first_generated_token_depends_on_depth_and_tail",
           0.0 if min(depth_effect, tail_effect) > 1e-6 else 1.0, tolerance=0.0,
           depth_logit_change=depth_effect, tail_logit_change=tail_effect)

    wrong = execute_online(model, tokens, prompt_end, schedule, wrong_backfill=True)
    wrong_error = max_error(wrong["logits"], reference["logits"])
    record("negative_control_final_exit_used_for_all_layer_backfills",
           0.0 if wrong_error > 1e-6 else 1.0, tolerance=0.0,
           incorrect_implementation_error=wrong_error,
           expected="The deliberately incorrect implementation must disagree.")

    grad = gradient_check()
    record("finite_difference_shared_lora_and_mixing_gradient",
           grad["max_absolute_error"], tolerance=1e-9, **grad)

    initial_grad = gradient_check(at_initialization=True)
    initial_gamma = max(abs(v) for v in initial_grad["analytic"][:3])
    initial_a = max(abs(v) for v in initial_grad["analytic"][3:6])
    initial_b = max(abs(v) for v in initial_grad["analytic"][6:])
    initial_error = max(initial_grad["max_absolute_error"], initial_a,
                        0.0 if min(initial_gamma, initial_b) > 1e-6 else 1.0)
    record("zero_gamma_zero_B_initialization_gradient_example", initial_error,
           tolerance=1e-9, gamma_gradient_max_abs=initial_gamma,
           A_gradient_max_abs=initial_a, B_gradient_max_abs=initial_b,
           note="A gradient vanishes at B=0. Nonzero gamma and B gradients are an example, not a universal guarantee.",
           **initial_grad)

    expected_cost = empty_cost()
    n = len(tokens)
    expected_cost.update(shared_layer_tokens=n,
                         actual_loop_layer_tokens=F_LAYERS * sum(a[0] for a in online["actions"]),
                         decision_layer_tokens=n,
                         tail_current_layer_tokens=TAIL_LAYERS * (n - prompt_end),
                         lm_readouts=n - prompt_end)
    # Count missing slots from the schedule alone, independent of cache code.
    for depth in range(2, 5):
        actual_positions = [t for t, action in enumerate(online["actions"])
                            if action[0] >= depth]
        if actual_positions:
            expected_cost["missing_loop_kv_slots"] += F_LAYERS * sum(
                online["actions"][s][0] < depth for s in range(max(actual_positions)))
    expected_cost["missing_loop_projection_matvecs"] = 2 * expected_cost["missing_loop_kv_slots"]
    for expert in range(3):
        positions = [t for t, action in enumerate(online["actions"])
                     if t >= prompt_end and action[1] == expert]
        if positions:
            expected_cost["tail_catchup_layer_tokens"] += TAIL_LAYERS * (
                max(positions) + 1 - len(positions))
    record("cost_includes_every_token_decision_and_tail_catchup",
           max_error(online["cost"], expected_cost), tolerance=0.0,
           observed=online["cost"], independently_counted=expected_cost,
           note="Operation categories are counted separately; this is not a FLOP or latency estimate.")

    return dict(schema_version=1, seed=SEED, width=WIDTH,
                loop_layers=F_LAYERS, tail_layers=TAIL_LAYERS,
                micro_model_experts=3, formal_architecture_experts=2,
                passed=all(item["passed"] for item in checks), checks=checks,
                scope="Deterministic micro-model algebra and causal cache semantics only; not Qwen production-executor validation.",
                qualifications=[
                    "Finite tests complement, and do not replace, the symbolic proof in the architecture document.",
                    "History completion defines a surrogate deep-round history; it is not full deep-prefix recomputation.",
                    "The controller uses scripted causal position biases to exercise mixed actions; policy learning is untested.",
                    "No Qwen weights, MoE kernels, training quality, GPU memory, or wall-clock speed are tested.",
                    "Before the final prompt position, only S, the first F round, and D execute; exits are saved and no tail or vocabulary head executes.",
                    "At the final prompt position the chosen tail first catches up all saved prompt exits, then produces the first generated-token logits.",
                    "The micro-model deliberately tests three tails and twelve actions, generalizing the formal architecture's two tails and eight actions.",
                    "Tail cursor catch-up can cost more than a single tail pass; controller computation is paid for every input token.",
                ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1]
                        / "docs" / "ARCHITECTURE_MATH_CHECK.v4.json")
    args = parser.parse_args()
    result = run_checks()
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(dict(passed=result["passed"], tests=len(result["checks"]),
                          output=str(args.output.resolve()),
                          errors={item["name"]: item["max_error"] for item in result["checks"]}),
                     ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
