"""GPU worker for one shard of one v3.3 stage. Launched by driver.py with CUDA_VISIBLE_DEVICES set.

    python -m experiments.chunk_depth_v33.worker <stage> --shard i --nshards n
Stages: A_tables, A_calcheck, A_online, B_features, B_train, B_eval, B_deploy_tables, B_deploy_online.
Document-level stages save partial results every 25 documents and resume from them.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import random
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import write_json
from experiments.shared_depth_v2.policy import lr_at
from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31 import screen as S
from experiments.chunk_depth_v31.features import extract
from experiments.chunk_depth_v32.tables import table
from experiments.chunk_depth_v32.stage1 import Head, head_policy
from experiments.chunk_depth_v33 import common as C
from experiments.chunk_depth_v33 import policies as P


def log(*args):
    print(time.strftime("%H:%M:%S"), *args, flush=True)


def model_with_lora(state=None):
    model = H.load_model()
    lora = S.Lora(model)
    if state is not None:
        load_lora(lora, state)
    return model, lora


def load_lora(lora, state):
    """state: path or list of tensors."""
    values = torch.load(state, weights_only=False)["lora"] if not isinstance(state, list) else state
    H.set_lora_enabled(lora.modules, True)
    with torch.no_grad():
        for p, v in zip(lora.params, values):
            p.copy_(v.to(p.device))


def run_docs(kind, items, shard, nshards, per_doc, scratch=False, every=25):
    if C.shard_is_done(kind, shard, nshards, scratch):
        log(kind, "shard already done")
        return
    pf = C.partial_file(kind, shard, nshards, scratch)
    rows = torch.load(pf, weights_only=False)["rows"] if pf.exists() else []
    done = {r[0] for r in rows}
    idx = C.shard_indices(len(items), shard, nshards)
    t0 = time.perf_counter()
    for i in idx:
        if i in done:
            continue
        rows.append((i, per_doc(i, items[i])))
        if len(rows) % every == 0:
            C.save_atomic({"rows": rows}, pf)
            rate = (time.perf_counter() - t0) / max(1, len(rows) - len(done))
            log(kind, f"{len(rows)}/{len(idx)}", f"{rate:.2f} s/doc")
    C.finish_shard(kind, shard, nshards, rows, scratch)
    log(kind, "shard finished", len(rows))


def stats_entry(item, nll, depths):
    s = R.doc_stats(item, nll, depths)
    return {"mean_nll": s["mean_nll"], "cost": s["cost"], "base16": s["base16"], "tokens": s["tokens"], "depths": s["depths"]}


def lock_policies():
    return json.loads((C.RESULTS / "policies_A.json").read_text(encoding="utf-8"))


def part_a_models():
    return (P.Ridge(torch.load(C.ART / "fit" / "R.pt", weights_only=False)),
            P.Ridge(torch.load(C.ART / "fit" / "H.pt", weights_only=False)))


# ----------------------------------------------------------------------------- experiment A

def a_tables(shard, nshards):
    model, _ = model_with_lora(str(C.THETA_S))
    for role in ("fit", "calibration"):
        items = C.load_role(role)
        run_docs(f"tables_{role}", items, shard, nshards, lambda i, it: R.retry(table, model, it), scratch=True)


def a_policy_list(lock, kappa, Rm, Hm, only_check=False):
    """Locked policies; within a family a parameter value chosen for several targets is executed once."""
    pols = []
    for fam, model in (("R", Rm), ("H", Hm)):
        seen = set()
        for budget in C.BUDGETS:
            entries = lock["policies"][fam][str(budget)]
            if only_check:
                if budget != 0.5:
                    continue
                entries = [e for e in entries if e["target"] in (0.495, 0.505)]
            for e in entries:
                if e["param"] in seen:
                    continue
                seen.add(e["param"])
                pols.append({"name": f"{fam}_B{budget}_t{e['target']:.3f}", "family": fam, "nu": e["param"],
                             "predicted_cost": e["predicted_cost"]})
    if not only_check:
        pols += [{"name": f"U_d{d}", "family": "U", "depth": d} for d in (6, 8, 12)]
    return pols


def a_calcheck(shard, nshards):
    lock, kappa = lock_policies(), P.kappas()
    Rm, Hm = part_a_models()
    pols = a_policy_list(lock, kappa, Rm, Hm, only_check=True)
    model, _ = model_with_lora(str(C.THETA_S))
    items = C.load_role("calibration")

    def per_doc(i, item):
        out = {}
        for p in pols:
            rule = P.r_policy(Rm, p["nu"], kappa) if p["family"] == "R" else P.h_policy(Hm, p["nu"], kappa)
            nll, depths = R.retry(R.online, model, item, rule)
            out[p["name"]] = stats_entry(item, nll, depths)
        return out
    run_docs("calcheck", items, shard, nshards, per_doc)


def a_online(shard, nshards):
    lock, kappa = lock_policies(), P.kappas()
    Rm, Hm = part_a_models()
    pols = a_policy_list(lock, kappa, Rm, Hm)
    model, _ = model_with_lora(str(C.THETA_S))
    items = C.load_role("confirm")

    def per_doc(i, item):
        tab = R.retry(table, model, item)
        out, agree = {}, {}
        for p in pols:
            if p["family"] == "R":
                rule = P.r_policy(Rm, p["nu"], kappa)
            elif p["family"] == "H":
                probe = {"reps": tab["reps"], "agree": [0, 0]}
                rule = P.h_policy(Hm, p["nu"], kappa, probe)
            else:
                rule = P.uniform_policy(p["depth"])
            nll, depths = R.retry(R.online, model, item, rule)
            out[p["name"]] = stats_entry(item, nll, depths)
            if p["family"] == "H":
                agree[p["name"]] = probe["agree"]
        return {"split": item["split"], "row": int(item["row"]), "part": C.confirm_part(i), "policies": out, "agree": agree}
    run_docs("online_A", items, shard, nshards, per_doc)


# ----------------------------------------------------------------------------- experiment B

def b_runs():
    return [(arm, seed) for seed in C.SEEDS for arm in C.ARMS]


def b_checkpoint(arm, seed):
    return C.ART / "b" / f"{arm}_s{seed}.pt"


def b_features(shard, nshards):
    model = H.load_model()
    items = C.load_role("btrain")
    run_docs("features_btrain", items, shard, nshards, lambda i, it: {"feats": R.retry(extract, model, it)["feats"]},
             scratch=True)


def b_train_one(arm, seed, items, feats):
    from experiments.chunk_depth_v32_followup import train as FT
    ckpt, log_path = b_checkpoint(arm, seed), C.RESULTS / "b" / f"train_{arm}_s{seed}.json"
    if ckpt.exists() and log_path.exists():
        log("B_train", arm, seed, "already done")
        return
    t_start = time.perf_counter()
    torch.manual_seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    head = FT.load_head()
    model, modules, params = FT.init_model()
    refs = FT.reference_logps(model, items, feats, head)
    if arm == "D":
        opts = [torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01), torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)]
    else:
        opts = [torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)]
    gen = torch.Generator(device="cpu").manual_seed(seed)
    order = torch.cat([torch.randperm(len(items), generator=gen) for _ in range(FT.EPOCHS)])
    planned = len(items) * FT.EPOCHS // FT.BATCH
    total = min(C.B_UPDATES, planned)
    logs = []
    for update in range(total):
        for opt in opts:
            for g_ in opt.param_groups:
                g_["lr"] = lr_at(update, total, 1e-4, 1e-5, 0.05)
        batch = order[update * FT.BATCH:(update + 1) * FT.BATCH].tolist()
        acc = [[torch.zeros_like(p) for p in params] for _ in range(2)]
        sums = {"gen": 0.0, "ctrl": 0.0, "depth8_nll": 0.0}
        before = [p.detach().clone() for p in params]
        for slot, j in enumerate(batch):
            g_loss, c_loss, stats = FT.loss_for_doc(model, items[j], feats[j], head, refs[j], FT.history_seed(seed, update, slot))
            if arm == "G":
                parts = [(0, g_loss)]
            elif arm == "C":
                parts = [(0, g_loss + c_loss)]
            else:
                parts = [(0, g_loss), (1, c_loss)]
            for n, (slot_acc, loss) in enumerate(parts):
                grads = torch.autograd.grad(loss / FT.BATCH, params, retain_graph=n + 1 < len(parts), allow_unused=True)
                for a, g in zip(acc[slot_acc], grads):
                    if g is not None:
                        a += g
            sums["gen"] += stats["component_values"]["gen"] / FT.BATCH
            sums["ctrl"] += stats["component_values"]["ctrl"] / FT.BATCH
            sums["depth8_nll"] += stats["depth8_nll"] / FT.BATCH
            del g_loss, c_loss, stats
        norms = []
        for opt, a in zip(opts, acc):
            for p, g in zip(params, a):
                p.grad = g
            norms.append(float(torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)))
            opt.step()
            for p in params:
                p.grad = None
        delta = math.sqrt(sum(float((p.detach() - b).double().square().sum()) for p, b in zip(params, before)))
        entry = {"update": update, "lr": opts[0].param_groups[0]["lr"], **sums, "grad_norms_before_clip": norms,
                 "parameter_delta_norm": delta}
        logs.append(entry)
        if update % 10 == 0 or update == total - 1:
            log("B_train", arm, seed, json.dumps(entry))
    C.save_atomic({"lora": [p.detach().cpu().clone() for p in params], "arm": arm, "seed": seed, "updates": total}, ckpt)
    write_json(log_path, C.v33_receipt("v33_b_training", arm=arm, seed=seed, updates=total, planned_updates=planned,
                                       documents=len(items), checkpoint_sha256=C.sha256_file(ckpt),
                                       seconds=time.perf_counter() - t_start, log=logs))
    del model, modules, params, opts
    torch.cuda.empty_cache()


def b_train(shard, nshards):
    items = C.load_role("btrain")
    feats = C.load_shards("features_btrain", 1, scratch=True)
    for n, (arm, seed) in enumerate(b_runs()):
        if n % nshards == shard:
            b_train_one(arm, seed, items, feats)


def b_eval(shard, nshards):
    names = [f"{arm}_s{seed}" for arm, seed in b_runs()]
    states = {f"{a}_s{s}": torch.load(b_checkpoint(a, s), weights_only=False)["lora"] for a, s in b_runs()}
    fixed = P.fixed_policy(P.Ridge(torch.load(C.R32, weights_only=False)), C.FIXED_NU)
    model, lora = model_with_lora()
    items = C.load_role("confirm")

    def per_doc(i, item):
        out = {}
        for n in names:
            load_lora(lora, states[n])
            nll, depths = R.retry(R.online, model, item, fixed)
            out[n] = stats_entry(item, nll, depths)
        return {"split": item["split"], "row": int(item["row"]), "part": C.confirm_part(i), "models": out}
    run_docs("eval_B", items, shard, nshards, per_doc)


def b_deploy_tables(shard, nshards):
    items = C.load_role("calibration")
    for n, arm in enumerate(C.ARMS):
        if n % nshards != shard:
            continue
        model, lora = model_with_lora(str(b_checkpoint(arm, 17)))
        run_docs(f"deploy_tables_{arm}", items, 0, 1, lambda i, it: R.retry(table, model, it), scratch=True)
        del model, lora
        torch.cuda.empty_cache()


def b_deploy_online(shard, nshards):
    lock = json.loads((C.RESULTS / "policies_B_deploy.json").read_text(encoding="utf-8"))
    head = Head(torch.load(C.HEAD32, weights_only=False))
    states = {arm: torch.load(b_checkpoint(arm, 17), weights_only=False)["lora"] for arm in C.ARMS}
    model, lora = model_with_lora()
    items = C.load_role("confirm_test")

    def per_doc(i, item):
        out = {}
        for arm in C.ARMS:
            load_lora(lora, states[arm])
            seen = set()
            for e in lock["policies"][arm]:
                if e["param"] in seen:
                    continue
                seen.add(e["param"])
                nll, depths = R.retry(R.online, model, item, head_policy(head, e["param"]))
                out[f"{arm}_t{e['target']:.3f}"] = stats_entry(item, nll, depths)
        return {"split": item["split"], "row": int(item["row"]), "policies": out}
    run_docs("deploy_online_B", items, shard, nshards, per_doc)


STAGES = {"A_tables": a_tables, "A_calcheck": a_calcheck, "A_online": a_online, "B_features": b_features,
          "B_train": b_train, "B_eval": b_eval, "B_deploy_tables": b_deploy_tables, "B_deploy_online": b_deploy_online}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=sorted(STAGES))
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--nshards", type=int, default=1)
    args = parser.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = False
    log("start", args.stage, args.shard, args.nshards, "smoke" if C.SMOKE else "full")
    STAGES[args.stage](args.shard, args.nshards)
    log("end", args.stage, args.shard)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
