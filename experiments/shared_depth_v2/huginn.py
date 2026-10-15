"""Huginn-0125 decomposition used by protocol jev-shared-depth-v2.

The native forward is: prelude -> recurrent core (raw state) -> ln_f -> coda -> ln_f -> lm_head.
This module exposes each stage so that the experiment can (a) cache ln_f(raw state) as the
coda input, (b) continue a depth-4 prefix to depth D from the raw state, and (c) share one
LoRA parameter set between the generation coda and the control coda.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import nn

from .common import MODEL_DIR
from .init_state import state_matrix

DEPTHS = (4, 8, 16, 32)
PREFIX = 128
WINDOW = 256
INPUTS = WINDOW - 1


def load_model(device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(str(MODEL_DIR), trust_remote_code=True, torch_dtype=dtype)
    model = model.to(device).eval()
    if getattr(model.config, "test_time_noise", 0) != 0:
        raise ValueError("test_time_noise must be 0")
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def freqs(model, positions: torch.Tensor) -> torch.Tensor:
    return model.freqs_cis.index_select(1, positions.to(model.freqs_cis.device))


def prelude(model, ids: torch.Tensor, fc: torch.Tensor, cache=None) -> torch.Tensor:
    e = model.transformer.wte(ids)
    if model.emb_scale != 1:
        e = e * model.emb_scale
    block_idx = torch.tensor(-1, dtype=torch.long)
    for block in model.transformer.prelude:
        block_idx += 1
        e = block(e, fc, block_idx, None, cache)
    return e


def core_steps(model, r, e, fc, start: int, stop: int, cache=None):
    """Run recurrent steps start..stop-1 from raw state r. Block index continues from 1+4*start."""
    block_idx = torch.tensor(1 + 4 * start, dtype=torch.long)
    for step in range(start, stop):
        r, block_idx = model.core_block_forward(r, e, fc, None, cache, block_idx, step)
    return r


def ln_f(model, x):
    return model.transformer.ln_f(x)


def coda(model, a, fc, cache=None):
    """a is the coda input ln_f(raw state); returns ln_f(coda(a)), the pre-head representation."""
    x = a
    block_idx = torch.tensor(0, dtype=torch.long)
    for block in model.transformer.coda:
        block_idx -= 1
        x = block(x, fc, block_idx, None, cache)
    return model.transformer.ln_f(x)


def head(model, z):
    return model.lm_head(z).float()


def initial_states(rows: list[int], positions: int, replica: int, device, dtype=torch.bfloat16):
    return torch.stack([state_matrix(r, positions, replica) for r in rows]).to(device=device, dtype=dtype)


@torch.no_grad()
def coda_inputs(model, ids: torch.Tensor, state: torch.Tensor, depths=DEPTHS) -> dict[int, torch.Tensor]:
    """Teacher-forcing features: ln_f(raw state) at each requested depth, no KV cache."""
    S = ids.shape[1]
    fc = freqs(model, torch.arange(S))
    e = prelude(model, ids, fc)
    out, r, done = {}, state, 0
    for d in sorted(depths):
        r = core_steps(model, r, e, fc, done, d)
        done = d
        out[d] = ln_f(model, r)
    return out


# ----------------------------------------------------------------------------- LoRA


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, generator: torch.Generator):
        super().__init__()
        self.base = base
        self.rank, self.scale = rank, alpha / rank
        bound = 1.0 / math.sqrt(base.in_features)
        a = torch.empty(rank, base.in_features, dtype=torch.float32).uniform_(-bound, bound, generator=generator)
        self.A = nn.Parameter(a.to(base.weight.device))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank, dtype=torch.float32, device=base.weight.device))
        self.enabled = True

    def forward(self, x):
        y = self.base(x)
        if not self.enabled:
            return y
        lora = nn.functional.linear(nn.functional.linear(x, self.A.to(x.dtype)), self.B.to(x.dtype))
        return y + self.scale * lora


LORA_TARGETS = (("attn", "Wqkv"), ("attn", "proj"), ("mlp", "fc"), ("mlp", "proj"))


def attach_lora(model, rank: int = 8, alpha: float = 16, seed: int = 0) -> list[LoRALinear]:
    """Wrap the four fused projections of both coda blocks, in a fixed order, seeded A init."""
    generator = torch.Generator(device="cpu").manual_seed(seed)
    modules = []
    for block in model.transformer.coda:
        for parent_name, child_name in LORA_TARGETS:
            parent = getattr(block, parent_name)
            base = getattr(parent, child_name)
            if isinstance(base, LoRALinear):
                raise ValueError("LoRA already attached")
            wrapped = LoRALinear(base, rank, alpha, generator)
            setattr(parent, child_name, wrapped)
            modules.append(wrapped)
    return modules


def detach_lora(model) -> None:
    for block in model.transformer.coda:
        for parent_name, child_name in LORA_TARGETS:
            parent = getattr(block, parent_name)
            mod = getattr(parent, child_name)
            if isinstance(mod, LoRALinear):
                setattr(parent, child_name, mod.base)


def lora_parameters(modules: list[LoRALinear]) -> list[nn.Parameter]:
    return [p for m in modules for p in (m.A, m.B)]


def set_lora_enabled(modules: list[LoRALinear], enabled: bool) -> None:
    for m in modules:
        m.enabled = enabled


# ----------------------------------------------------------------------------- deployment path


@dataclass
class PlanResult:
    control_probs: torch.Tensor
    logits: torch.Tensor          # [T, vocab] float32 for output positions 127.. in order
    depth: int


def new_cache():
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    cls = get_class_from_dynamic_module("raven_modeling_minimal.HuginnDynamicCache", str(MODEL_DIR))
    return cls(lookup_strategy="full")


@torch.no_grad()
def plan_request(model, ids: torch.Tensor, row: int, replica: int, depth: int | None, readout=None,
                 decode_ids: torch.Tensor | None = None, steps_out: int = 0, native: bool = False):
    """Execute the shared-control path with a KV cache (protocol section 3.2).

    ids: [1, 128] prefix. If depth is None the readout chooses argmax. decode_ids: optional
    teacher-forced continuation tokens [1, n] fed one at a time (positions P..). Otherwise
    greedily generate steps_out tokens. native=True runs the controller-free path: the prefix
    goes directly to depth D with one coda pass. Returns PlanResult with logits for positions
    P-1.. in order.
    """
    device = ids.device
    cache = new_cache()
    P = ids.shape[1]
    fc = freqs(model, torch.arange(P))
    e = prelude(model, ids, fc, cache)
    r0 = initial_states([row], INPUTS, replica, device, e.dtype)
    probs = None
    if native:
        r = core_steps(model, r0[:, :P], e, fc, 0, depth, cache)
        z = coda(model, ln_f(model, r), fc, cache)
    else:
        r = core_steps(model, r0[:, :P], e, fc, 0, 4, cache)
        z = coda(model, ln_f(model, r), fc, cache)
        if readout is not None:
            probs = torch.softmax(readout(z[:, -1].float()), -1)[0]
        if depth is None:
            depth = DEPTHS[int(probs.argmax())]
        if depth > 4:
            r = core_steps(model, r, e, fc, 4, depth, cache)
            z = coda(model, ln_f(model, r), fc, cache)   # rebuilds coda KV at negative indices
    logits = [head(model, z[:, -1])[0]]
    n = decode_ids.shape[1] if decode_ids is not None else steps_out
    token = None
    for k in range(n):
        t = P + k
        token = decode_ids[:, k:k + 1] if decode_ids is not None else token
        if token is None:
            token = logits[-1].argmax().view(1, 1)
        fc_t = freqs(model, torch.tensor([t]))
        e_t = prelude(model, token, fc_t, cache)
        r_t = core_steps(model, r0[:, t:t + 1], e_t, fc_t, 0, depth, cache)
        z_t = coda(model, ln_f(model, r_t), fc_t, cache)
        logits.append(head(model, z_t[:, -1])[0])
        if decode_ids is None:
            token = logits[-1].argmax().view(1, 1)
    return PlanResult(control_probs=probs, logits=torch.stack(logits), depth=depth), cache
