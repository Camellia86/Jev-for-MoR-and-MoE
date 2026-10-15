"""Protocol sections 4.3-4.5: train one arm (E, J, F16-A, FMIX-A) of the shared-coda LoRA.

Per update (8 documents, microbatch 1, one optimizer step):
  1. no-grad control path -> mu (fixed for this document's generation weights)
  2. for each depth: coda + head on cached features, backward of weight_d * NLL_d / 8
  3. E/J only: re-run control path with grad; backward of (route surrogate + auxiliary) / 8
  4. clip to norm 1, AdamW step with warmup + cosine schedule
"""

from __future__ import annotations

import argparse
import json
import time

import torch

from . import cost, policy
from . import huginn as H
from .common import ARTIFACTS, RESULTS, Timer, load_config, receipt, sha256_file, write_json
from .extract_features import load_features
from .warm import new_readout

ARMS = ("E", "J", "F16-A", "FMIX-A")


def load_readout(lam: float | None = None):
    blob = torch.load(ARTIFACTS / "pilot" / "readouts.pt")
    lam = blob["selected_lambda"] if lam is None else lam
    lin = new_readout()
    lin.load_state_dict(blob["readouts"][lam])
    for p in lin.parameters():
        p.requires_grad_(False)
    return lin.cuda().eval(), lam


def generation_weights(arm: str, mu: torch.Tensor) -> torch.Tensor:
    if arm in ("E", "J"):
        return mu.detach().float()
    if arm == "F16-A":
        return torch.tensor([0.0, 0.0, 1.0, 0.0], device=mu.device)
    return torch.full((4,), 0.25, device=mu.device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--role", default="pilot_train")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--tag", default="pilot")
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    cfg = load_config()
    opt_cfg = cfg["optimizer"]
    costs: dict = {}
    win = torch.load(ARTIFACTS / "data" / "windows.pt")
    header = json.loads((ARTIFACTS / "features" / f"{args.role}.json").read_text(encoding="utf-8"))
    n_docs = len(header["documents"])
    batch = opt_cfg["effective_batch"]
    total = n_docs * args.epochs // batch

    with Timer(costs, "model_load_seconds"):
        model = H.load_model()
    modules = H.attach_lora(model, cfg["lora"]["rank"], cfg["lora"]["alpha"], seed=args.seed)
    params = H.lora_parameters(modules)
    optimizer = torch.optim.AdamW(params, lr=opt_cfg["lr"], weight_decay=opt_cfg["weight_decay"])
    readout, lam = load_readout()
    norm = cost.normalized()
    f_plan = torch.tensor([norm["f_plan"][d] for d in H.DEPTHS], dtype=torch.float64, device="cuda")
    z0 = torch.load(ARTIFACTS / "pilot" / f"control_rep_zero_{args.role}.pt").cuda()
    with torch.no_grad():
        mu_ref_all = policy.mean_policy(readout(z0))
    fc, fcp = H.freqs(model, torch.arange(255)), H.freqs(model, torch.arange(128))
    gen = torch.Generator(device="cpu").manual_seed(args.seed)
    order = torch.cat([torch.randperm(n_docs, generator=gen) for _ in range(args.epochs)])

    out_dir = ARTIFACTS / args.tag / f"{args.arm}_s{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    log = []
    checkpoints = {int(round(total * q)) for q in (0.25, 0.5, 0.75, 1.0)}
    start = time.perf_counter()
    for update in range(total):
        t0 = time.perf_counter()
        for g in optimizer.param_groups:
            g["lr"] = policy.lr_at(update, total, opt_cfg["lr"], opt_cfg["end_lr"], opt_cfg["warmup_fraction"])
        optimizer.zero_grad(set_to_none=True)
        stats = {"objective": 0.0, "aux": 0.0, "nll": torch.zeros(4), "mu": torch.zeros(4)}
        for k in range(batch):
            j = int(order[update * batch + k])
            feats = load_features(args.role, j, header)
            midx = header["manifest_indices"][j]
            targets = win[midx, 128:256].long().cuda()
            prefix = feats["prefix"].cuda()[None]
            with torch.no_grad():
                mu0 = policy.mean_policy(readout(H.coda(model, prefix, fcp)[0, 127].float()))
            weights = generation_weights(args.arm, mu0)
            costs_d = torch.zeros(4, dtype=torch.float64, device="cuda")
            for d_i, depth in enumerate(H.DEPTHS):
                if args.arm == "F16-A" and depth != 16:
                    continue
                logits = H.head(model, H.coda(model, feats[depth].cuda()[None], fc))[0, 127:]
                nll = torch.nn.functional.cross_entropy(logits, targets)
                (weights[d_i] * nll / batch).backward()
                costs_d[d_i] = nll.detach().double() + lam * f_plan[d_i]
                stats["nll"][d_i] += float(nll) / batch
            if args.arm in ("E", "J"):
                z = H.coda(model, prefix, fcp)[0, 127].float()
                logit = readout(z)
                alpha = policy.dirichlet_alpha(logit)
                mu = alpha / alpha.sum()
                aux = policy.auxiliary(mu, costs_d, mu_ref_all[j])
                if args.arm == "E":
                    route = policy.route_exact(mu, costs_d)
                else:
                    route, _ = policy.route_jev(alpha, costs_d, seed=args.seed * 10**9 + update * 1000 + k)
                ((route + aux) / batch).backward()
                with torch.no_grad():
                    stats["objective"] += float((mu * costs_d).sum() + aux) / batch
                    stats["aux"] += float(aux) / batch
                    stats["mu"] += mu.detach().float().cpu() / batch
            else:
                stats["mu"] += mu0.float().cpu() / batch
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, opt_cfg["clip_grad_norm"]))
        optimizer.step()
        torch.cuda.synchronize()
        entry = {"update": update, "lr": optimizer.param_groups[0]["lr"], "grad_norm_preclip": grad_norm,
                 "objective": stats["objective"], "aux": stats["aux"],
                 "nll": [round(float(v), 5) for v in stats["nll"]], "mu": [round(float(v), 4) for v in stats["mu"]],
                 "seconds": time.perf_counter() - t0}
        log.append(entry)
        if update % 8 == 0 or update == total - 1:
            print(json.dumps(entry), flush=True)
        if update + 1 in checkpoints:
            torch.save({"lora": [p.detach().cpu() for p in params], "update": update + 1, "arm": args.arm,
                        "seed": args.seed, "lambda": lam}, out_dir / f"lora_u{update + 1}.pt")
    costs["train_seconds"] = time.perf_counter() - start
    final = out_dir / f"lora_u{total}.pt"
    payload = receipt("arm_training", protocol_revision="2.1", tag=args.tag, arm=args.arm, seed=args.seed,
                      role=args.role, epochs=args.epochs, updates=total, lambda_=lam, documents=n_docs,
                      lora_init="A ~ U(+-1/sqrt(d_in)) from CPU generator seeded by training seed in coda0..1 x "
                                "(Wqkv, attn.proj, mlp.fc, mlp.proj) order; B = 0",
                      final_checkpoint=str(final), final_checkpoint_sha256=sha256_file(final),
                      peak_allocated_gib=torch.cuda.max_memory_allocated() / 2**30, log=log, costs=costs)
    write_json(RESULTS / f"{args.tag}_train_{args.arm}_s{args.seed}.json", payload)
    print(json.dumps({"arm": args.arm, "seed": args.seed, "updates": total, "costs": costs,
                      "seconds_per_update": costs["train_seconds"] / total}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
