"""v3.2 停止后的机制诊断：从同一 theta_S 继续训练 E-G / E-C。

这是新的诊断初始化，不是原协议获准执行的 stage 2。两臂只在是否反传
控制项上不同；各自策略更新后的历史和生成权重允许不同。
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import sha256_file
from experiments.shared_depth_v2.policy import lr_at
from experiments.chunk_depth_v31 import train as T31
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v32 import common as C32
from experiments.chunk_depth_v32_followup import common as C

K, BATCH, EPOCHS, SEED = 5, 8, 2, 17
SNAPSHOTS = (0, 23, 46, 92)


class FrozenHead:
    """固定 FP64 读出；保留到输入表示的梯度。"""

    def __init__(self, blob, device="cuda"):
        self.W, self.beta, self.m, self.sd = [
            blob[k].detach().to(device=device, dtype=torch.float64)
            for k in ("W", "beta", "m", "sd")
        ]
        self.lam, self.tau = float(blob["lambda"]), float(blob["tau"])
        if self.lam != 1.0 or self.tau != 0.05:
            raise ValueError("诊断固定 lambda=1、tau=0.05，读出与声明不符")
        if self.W.shape != (K, 5290) or self.m.shape != (5280,):
            raise ValueError("读出形状与 5 动作、5280+10 维输入不符")
        if not all(torch.isfinite(x).all() for x in (self.W, self.beta, self.m, self.sd)):
            raise ValueError("读出包含非有限值")
        if not bool((self.sd > 0).all()):
            raise ValueError("标准差必须为正")
        self.kappa_s, self.kappa_e = T31.load_kv()

    @property
    def device(self):
        return self.W.device

    def logp(self, z, phi):
        phi = torch.as_tensor(phi, dtype=torch.float64, device=self.device)
        x = torch.cat([(z.to(self.device).double() - self.m) / self.sd, phi], -1)
        return torch.log_softmax(x @ self.W.T + self.beta, -1)


def load_head(device="cuda"):
    return FrozenHead(torch.load(C32.OUT / "head.pt", map_location="cpu", weights_only=False), device)


def init_model(seed=SEED):
    """加载共同非零 LoRA；两个臂均重新初始化同配置的 AdamW。"""
    C32.check_theta_s()
    model = H.load_model()
    modules = H.attach_lora(model, 8, 16, seed=seed)
    params = H.lora_parameters(modules)
    saved = torch.load(C32.THETA_S, map_location="cpu", weights_only=False)["lora"]
    if len(saved) != len(params):
        raise ValueError("theta_S 参数数目不符")
    with torch.no_grad():
        for p, v in zip(params, saved):
            if p.shape != v.shape or p.dtype != torch.float32:
                raise ValueError("theta_S 参数形状或 FP32 主参数不符")
            p.copy_(v.to(p.device, dtype=p.dtype))
        if not any(bool(torch.count_nonzero(p)) for p in params[1::2]):
            raise ValueError("theta_S 的 LoRA B 全为零；诊断初始化错误")
    return model, modules, params


def history_seed(seed, update, slot):
    """历史采样与数据打乱独立；两臂使用同一派生规则。"""
    key = f"chunk-v32-followup-history:{seed}:{update}:{slot}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "big") % (2**63)


@torch.no_grad()
def reference_logps(model, items, feats, head, check=None):
    """共同 theta_S 下每篇 [前一行深度, 行, 动作] 的固定参考。"""
    refs = []
    for item, f in zip(items, feats):
        if check is not None:
            check()
        st = T31.doc_structure(item)
        fc = H.freqs(model, torch.arange(len(item["ids"])))
        phi = torch.tensor([l["phi"] for l in st], device=head.device, dtype=torch.float64)
        boundary = [l["a"] - 1 for l in st]
        parts = []
        for k in range(K):
            z = H.coda(model, f["feats"][k].to(head.device)[None], fc)[0]
            parts.append(head.logp(z[boundary], phi).cpu())
        refs.append(torch.stack(parts))
    return refs


def loss_for_doc(model, item, f, head, reference_logp, seed):
    """返回按文档 token 数归一化的生成项、控制项及可审计分项。

    ctrl = E_p[C] + 0.1 CE(s,p) + 0.01 Brier(s,p) + 0.1 KL(p||p_ref)。
    成本、软标签和采样历史停止梯度；读出输入保留梯度。该目标是局部代理。
    stats['component_losses'] 保留计算图，供独立的分项梯度诊断使用。
    """
    device = head.device
    st = T31.doc_structure(item)
    N = sum(l["n"] for l in st)
    if N <= 0:
        raise ValueError("文档没有计分 token")
    ids = torch.tensor(item["ids"], device=device)
    fc = H.freqs(model, torch.arange(len(ids)))
    zs, nll = [], []
    for k in range(K):
        z = H.coda(model, f["feats"][k].to(device)[None], fc)[0]
        zs.append(z)
        nll.append(torch.nn.functional.cross_entropy(
            H.head(model, z[None])[0, :-1], ids[1:], reduction="none"))
    zero = torch.zeros((), dtype=torch.float64, device=device)
    gen = zero
    parts = {k: zero for k in ("route", "ce", "brier", "kl")}
    token_weights = torch.zeros(K, dtype=torch.float64, device=device)
    sampled_tokens = torch.zeros(K, dtype=torch.float64, device=device)
    sampled_lines = torch.zeros(K, dtype=torch.float64, device=device)
    depth8_nll = zero
    hist, kprev, sampled, logps = [], R.V3.index(16), [], []
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    for i, l in enumerate(st):
        logp = head.logp(zs[kprev][l["a"] - 1], l["phi"])
        p = logp.exp()
        # 单独 CPU RNG 流，不消耗数据顺序或模型全局 RNG。
        k = int(torch.multinomial(p.detach().cpu(), 1, generator=generator))
        sampled.append(k)
        logps.append(logp.detach().cpu())
        sampled_lines[k] += 1
        if l["n"]:
            Lc = torch.stack([v[l["scored"]].sum() for v in nll])
            coverage = [R.coverage(hist, d) for d in R.V3]
            kv = torch.tensor([head.kappa_s * s + head.kappa_e * e for s, e in coverage],
                              dtype=torch.float64, device=device)
            cost = Lc.detach().double() + l["n"] * (head.lam * R.R_REL.to(device) + kv)
            s = torch.softmax(-cost / (l["n"] * head.tau), -1).detach()
            ref = reference_logp[kprev, i].to(device=device, dtype=torch.float64).detach()
            gen = gen + (p.detach() * Lc).sum() / N
            parts["route"] = parts["route"] + (p * cost).sum() / N
            parts["ce"] = parts["ce"] - (l["n"] / N) * (s * logp).sum()
            parts["brier"] = parts["brier"] + (l["n"] / N) * ((p - s) ** 2).sum()
            parts["kl"] = parts["kl"] + (l["n"] / N) * (p * (logp - ref)).sum()
            token_weights += l["n"] * p.detach()
            sampled_tokens[k] += l["n"]
            depth8_nll = depth8_nll + Lc[2].detach().double() / N
        hist.append((l["b"] - l["a"], R.V3[k]))
        kprev = k
    weighted = {"route": parts["route"], "ce": 0.1 * parts["ce"],
                "brier": 0.01 * parts["brier"], "kl": 0.1 * parts["kl"]}
    ctrl = sum(weighted.values())
    return gen, ctrl, {
        "n_tokens": N, "n_lines": len(st), "depth8_nll": float(depth8_nll),
        "generation_token_weights": (token_weights / N).cpu().tolist(),
        "generation_token_mass": token_weights.cpu().tolist(),
        "sampled_token_mass": sampled_tokens.cpu().tolist(),
        "sampled_line_counts": sampled_lines.cpu().tolist(),
        "sampled_actions": sampled, "policy_logp": torch.stack(logps),
        "component_losses": {"gen": gen, **weighted},
        "component_values": {"gen": float(gen.detach()), "ctrl": float(ctrl.detach()),
                             **{k: float(v.detach()) for k, v in parts.items()}},
    }


def rng_state(data_generator=None):
    return {"torch_cpu": torch.get_rng_state(), "torch_cuda": torch.cuda.get_rng_state_all(),
            "python": random.getstate(),
            "data_generator": None if data_generator is None else data_generator.get_state()}


def checkpoint(params, opt, arm, seed, update, order, data_generator, identities, **extra):
    return {"lora": [p.detach().cpu().clone() for p in params], "opt": opt.state_dict(),
            "rng": rng_state(data_generator), "arm": arm, "seed": seed, "update": update,
            "data_order": order.clone(), "identities": identities,
            "initialization": "existing theta_S, fresh AdamW; post-stage1-stop diagnostic", **extra}


def run(arm, smoke=False):
    started = time.perf_counter()
    cap = float(C.CONFIG.get("train_cap_seconds", 7200))
    guard = lambda: C.check_time(started, cap)
    identities = C.verify_inputs()
    C.OUT.mkdir(parents=True, exist_ok=True)
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.manual_seed(SEED)
    random.seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    items = torch.load(T31.OUT / "items_pilot_train.pt", map_location="cpu", weights_only=False)
    feats = torch.load(T31.OUT / "features_pilot_train.pt", map_location="cpu", weights_only=False)
    training_source_sha256 = {name: sha256_file(T31.OUT / name)
                              for name in ("items_pilot_train.pt", "features_pilot_train.pt")}
    if len(items) != 373 or len(feats) != len(items):
        raise ValueError("诊断固定使用 373 篇 pilot_train 和一一对应的缓存")
    head = load_head()
    model, modules, params = init_model()
    refs = reference_logps(model, items, feats, head, check=guard)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    data_generator = torch.Generator(device="cpu").manual_seed(SEED)
    order = torch.cat([torch.randperm(len(items), generator=data_generator) for _ in range(EPOCHS)])
    planned = len(items) * EPOCHS // BATCH
    if planned != 93:
        raise ValueError("诊断要求 93 次更新")
    total = 1 if smoke else planned
    prefix = "smoke" if smoke else "pilot"
    stem = f"{prefix}_{arm}_s{SEED}"
    final_path = C.OUT / f"{stem}.pt"
    report_path = C.RESULTS / f"{stem}.json"
    if final_path.exists() or report_path.exists():
        raise FileExistsError(f"输出已存在，拒绝覆盖：{stem}")
    reference_path = C.OUT / f"{stem}_reference_logp.pt"
    torch.save({"logp": refs, "identities": identities,
                "source": "theta_S before diagnostic updates"}, reference_path)
    logs = []
    for update in range(total):
        guard()
        for group in opt.param_groups:
            group["lr"] = lr_at(update, planned, 1e-4, 1e-5, 0.05)
        batch_indices = order[update * BATCH:(update + 1) * BATCH].tolist()
        if arm == "E-G" and update in SNAPSHOTS:
            name = f"snapshot_{stem}_u{update:03d}.pt"
            torch.save(checkpoint(params, opt, arm, SEED, update, order, data_generator, identities,
                                  batch_indices=batch_indices, reference_path=str(reference_path)), C.OUT / name)
        opt.zero_grad(set_to_none=True)
        before = [p.detach().clone() for p in params]
        sums = {k: 0.0 for k in ("gen", "ctrl", "route", "ce", "brier", "kl", "depth8_nll")}
        train_weights = torch.zeros(K, dtype=torch.float64)
        token_mass, sampled_mass, line_counts = [torch.zeros(K, dtype=torch.float64) for _ in range(3)]
        n_tokens, n_lines = 0, 0
        step_start = time.perf_counter()
        for slot, j in enumerate(batch_indices):
            gen, ctrl, stats = loss_for_doc(model, items[j], feats[j], head, refs[j], history_seed(SEED, update, slot))
            loss = (gen + (ctrl if arm == "E-C" else 0)) / BATCH
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError(f"非有限损失：{arm}, update={update}, doc={j}")
            loss.backward()
            for name, value in stats["component_values"].items():
                sums[name] += value / BATCH
            sums["depth8_nll"] += stats["depth8_nll"] / BATCH
            train_weights += torch.tensor(stats["generation_token_weights"]) / BATCH
            token_mass += torch.tensor(stats["generation_token_mass"])
            sampled_mass += torch.tensor(stats["sampled_token_mass"])
            line_counts += torch.tensor(stats["sampled_line_counts"])
            n_tokens += stats["n_tokens"]
            n_lines += stats["n_lines"]
            del gen, ctrl, loss, stats
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True))
        opt.step()
        delta_sq = sum(float((p.detach() - old).double().square().sum()) for p, old in zip(params, before))
        param_sq = sum(float(old.double().square().sum()) for old in before)
        delta = math.sqrt(delta_sq)
        entry = {"update": update, "batch_indices": batch_indices, "lr": opt.param_groups[0]["lr"],
                 **sums, "optimized_loss": sums["gen"] + (sums["ctrl"] if arm == "E-C" else 0),
                 "grad_norm_before_clip": grad_norm, "gradient_clip_coefficient": min(1.0, 1.0 / (grad_norm + 1e-6)),
                 "actual_parameter_delta_norm": delta, "parameter_norm_before": math.sqrt(param_sq),
                 "relative_parameter_delta_norm": delta / max(math.sqrt(param_sq), 1e-30),
                 "generation_weights_document_normalized": train_weights.tolist(),
                 "generation_weights_token_pooled": (token_mass / n_tokens).tolist(),
                 "sampled_weights_token_pooled": (sampled_mass / n_tokens).tolist(),
                 "sampled_weights_line_pooled": (line_counts / n_lines).tolist(),
                 "n_scored_tokens": n_tokens, "n_lines": n_lines,
                 "seconds": time.perf_counter() - step_start}
        logs.append(entry)
        print(json.dumps(entry), flush=True)
        del before
        guard()
    torch.save(checkpoint(params, opt, arm, SEED, total, order, data_generator, identities,
                          reference_path=str(reference_path), smoke=smoke), final_path)
    C.write_json(report_path, C.record("mechanism_training", arm=arm, seed=SEED, smoke=smoke,
        updates=total, planned_updates=planned, epochs=EPOCHS, batch_size=BATCH,
        documents=len(items), unused_order_entries=len(order) - planned * BATCH,
        initialization="theta_S existing LoRA; fresh AdamW", head_frozen=True, lambda_=head.lam, tau=head.tau,
        control_coefficients={"ce": 0.1, "brier": 0.01, "kl": 0.1},
        kappa_s=head.kappa_s, kappa_e=head.kappa_e,
        kv_source_sha256=sha256_file(T31.RESULTS / "kv.json"),
        kv_precision="Existing kv.json full precision; config's short decimals are rounded descriptions",
        training_source_sha256=training_source_sha256,
        identities=identities, reference_logp=str(reference_path), checkpoint=str(final_path),
        checkpoint_sha256=sha256_file(final_path), seconds=time.perf_counter() - started,
        cap_seconds=cap, log=logs))
    print(json.dumps({"checkpoint": str(final_path), "report": str(report_path)}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("E-G", "E-C"), required=True)
    parser.add_argument("--smoke", action="store_true", help="只运行一个更新，使用独立 smoke 文件名")
    args = parser.parse_args()
    run(args.arm, args.smoke)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
