"""Pilot step 4 (plan v3.1 sections 4.1-4.5): readout warm-up with the lambda rule, then arm training.

Usage:
  python -m experiments.chunk_depth_v31.train warm
  python -m experiments.chunk_depth_v31.train arm --arm E --seed 17
Controls use K = 5 actions (depths 4, 6, 8, 12, 16). The training objective is the local surrogate of
plan section 4.3: the current line's cost with the history sampled from the current policy without
gradient; the KV coverage correction uses kappa_s, kappa_e from results/chunk-depth-v31/kv.json.
"""

from __future__ import annotations

import argparse
import json
import math
import time

import torch

from experiments.shared_depth_v2 import huginn as H
from experiments.shared_depth_v2.common import receipt, sha256_file, write_json
from experiments.shared_depth_v2.policy import dirichlet_log_prob, lr_at, sample_dirichlet
from experiments.chunk_depth_v31.data import OUT, RESULTS
from experiments.chunk_depth_v31 import run as R

K = 5
KAPPA_D, EPS, ETA, TAU = 256.0, 0.01, 0.001, 0.05
WARM_SEED = 20261011
LAMBDA_GRID = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)
BUDGET = 0.5
IN_DIM = 5280 + 10


# ----------------------------------------------------------------------------- policy maths (K = 5)

def alpha_of(logits):
    p = torch.softmax(logits.double(), -1)
    return KAPPA_D * ((1 - EPS) * p + EPS / K) + ETA


def mean_policy(logits):
    a = alpha_of(logits)
    return a / a.sum(-1, keepdim=True)


def aux_loss(mu, cost_per_token, mu_ref):
    s = torch.softmax(-cost_per_token.detach().double() / TAU, -1)
    ce = -(s * mu.log()).sum(-1)
    brier = ((mu - s) ** 2).sum(-1)
    kl = (mu * (mu.log() - mu_ref.double().log())).sum(-1)
    return 0.1 * ce + 0.01 * brier + 0.1 * kl


def route_jev(alpha, cost, seed, m=64):
    q = sample_dirichlet(alpha, m, seed).to(alpha.device)
    v = q @ cost.detach().double()
    adv = (v - (v.sum() - v) / (m - 1)).detach()
    return (adv * dirichlet_log_prob(q, alpha)).mean()


def new_readout():
    g = torch.Generator(device="cpu").manual_seed(WARM_SEED)
    lin = torch.nn.Linear(IN_DIM, K)
    bound = 1 / math.sqrt(IN_DIM)
    with torch.no_grad():
        lin.weight.copy_(torch.empty(K, IN_DIM).uniform_(-bound, bound, generator=g))
        lin.bias.zero_()
    return lin


def load_kv():
    kv = json.loads((RESULTS / "kv.json").read_text(encoding="utf-8"))
    return kv["kappa_s"], kv["kappa_e"]


# ----------------------------------------------------------------------------- document structure

def doc_structure(item):
    q_end, lines, S = R.lines_of(item)
    out = []
    prev_len = q_end
    for c, (a, b) in enumerate(lines, start=1):
        sc = R.scored_in(item, a, b)
        out.append({"c": c, "a": a, "b": b, "scored": sc, "n": len(sc), "phi": R.pos_feats(c, prev_len),
                    "f": torch.tensor([R.line_cost(a, b, d) for d in R.V3], dtype=torch.float64)})
        prev_len = b - a
    return out


@torch.no_grad()
def frozen_boundary_reps(model, items, feats):
    """For every document: z at each line's decision position p_{c-1} under each depth (frozen model)."""
    out = []
    for item, f in zip(items, feats):
        S = len(item["ids"])
        fc = H.freqs(model, torch.arange(S))
        st = doc_structure(item)
        zs = torch.stack([H.coda(model, f["feats"][k].cuda()[None], fc)[0][[l["a"] - 1 for l in st]].float().cpu()
                          for k in range(K)])                       # [K, lines, 5280]
        out.append(zs)
    return out


# ----------------------------------------------------------------------------- warm-up

def warm_rows(items, feats, reps, lam, kappa_s, kappa_e):
    """Rows (x, cost per token, n): history uniform at d_prev (first line: question at 16, empty history)."""
    xs, cs = [], []
    for item, f, zs in zip(items, feats, reps):
        st = doc_structure(item)
        for i, l in enumerate(st):
            if l["n"] == 0:
                continue
            L = f["nll"][:, l["scored"]].double().sum(1)
            for dprev in (R.V3 if i > 0 else (16,)):
                kprev = R.V3.index(dprev)
                hist = [(st[j]["b"] - st[j]["a"], dprev) for j in range(i)]
                kv = torch.tensor([kappa_s * R.coverage(hist, d)[0] + kappa_e * R.coverage(hist, d)[1] for d in R.V3],
                                  dtype=torch.float64)
                cost = (L + lam * l["n"] * R.R_REL + l["n"] * kv) / l["n"]
                xs.append(torch.cat([zs[kprev, i].double(), torch.tensor(l["phi"], dtype=torch.float64)]))
                cs.append(cost)
    return torch.stack(xs), torch.stack(cs)


def train_readout(x, c):
    lin = new_readout().cuda().double()
    opt = torch.optim.AdamW(lin.parameters(), lr=0.003, weight_decay=0.01)
    g = torch.Generator(device="cpu").manual_seed(WARM_SEED)
    x, c = x.cuda(), c.cuda()
    uniform = torch.full((K,), 1.0 / K, dtype=torch.float64, device="cuda")
    n, bs, pos = len(x), 32, 0
    order = torch.randperm(n, generator=g)
    for _ in range(400):
        if pos + bs > n:
            order, pos = torch.randperm(n, generator=g), 0
        idx = order[pos:pos + bs].cuda()
        pos += bs
        mu = mean_policy(lin(x[idx]))
        loss = ((mu * c[idx]).sum(-1) + aux_loss(mu, c[idx], uniform.expand_as(mu))).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return lin.float().cpu()


def greedy_cost(lin, items, reps):
    """Uncalibrated (b = 0) sequential policy on table representations: mean relative cost."""
    num = den = 0.0
    with torch.no_grad():
        for item, zs in zip(items, reps):
            st = doc_structure(item)
            dprev = 16
            for i, l in enumerate(st):
                x = torch.cat([zs[R.V3.index(dprev), i], torch.tensor(l["phi"])])[None]
                k = int(mean_policy(lin(x))[0].argmax())
                num += float(l["f"][k])
                den += float(l["f"][R.V3.index(16)])
                dprev = R.V3[k]
    return num / den


def warm_main():
    torch.backends.cuda.matmul.allow_tf32 = False
    kappa_s, kappa_e = load_kv()
    items = torch.load(OUT / "items_warm.pt")
    feats = torch.load(OUT / "features_warm.pt")
    model = H.load_model()
    reps = frozen_boundary_reps(model, items, feats)
    rows, readouts = [], {}
    for lam in LAMBDA_GRID:
        x, c = warm_rows(items, feats, reps, lam, kappa_s, kappa_e)
        lin = train_readout(x, c)
        b = greedy_cost(lin, items, reps)
        readouts[lam] = lin.state_dict()
        rows.append({"lambda": lam, "rows": len(x), "greedy_relative_cost": b, "abs_log_ratio": abs(math.log(b / BUDGET))})
        print(json.dumps(rows[-1]), flush=True)
    best = min(rows, key=lambda r: (round(r["abs_log_ratio"], 12), r["lambda"]))
    torch.save({"readouts": readouts, "selected_lambda": best["lambda"]}, OUT / "readouts.pt")
    # SCHED-A schedule: position ridge on warm frozen tables, nu from warm table cost closest to the budget
    from experiments.chunk_depth_v3.phase0_online import Ridge
    px, py = [], []
    for item, f in zip(items, feats):
        for l in doc_structure(item):
            if l["n"]:
                px.append(l["phi"])
                py.append(f["nll"][:, l["scored"]].double().mean(1))
    px, py = torch.tensor(px, dtype=torch.float64), torch.stack(py)
    ridge = Ridge(px, py, 10.0)
    all_lines = [l for item in items for l in doc_structure(item)]
    preds = ridge(torch.tensor([l["phi"] for l in all_lines], dtype=torch.float64))
    fmat = torch.stack([l["f"] for l in all_lines])
    den = float(fmat[:, 4].sum())
    best_nu = None
    for nu in [i * 0.05 for i in range(0, 161)]:
        ks = (preds + nu * R.R_REL).argmin(1)
        num = float(fmat[torch.arange(len(ks)), ks].sum())
        if num / den <= BUDGET:
            best_nu = (nu, num / den)
            break
    torch.save({"w": ridge.w, "mu": ridge.mu, "sd": ridge.sd, "nu": best_nu[0]}, OUT / "sched_a_schedule.pt")
    write_json(RESULTS / "warm.json", receipt("chunk_depth_v3_1_pilot_warm", kappa_s=kappa_s, kappa_e=kappa_e,
               rows=rows, selected_lambda=best["lambda"], sched_a={"nu": best_nu[0], "warm_table_cost": best_nu[1],
               "note": "pilot simplification: SCHED-A schedule nu chosen on warm table cost, not by real execution"}))
    print(json.dumps({"selected_lambda": best["lambda"], "sched_a": best_nu}))


# ----------------------------------------------------------------------------- arm training

def sched_depth(sched, phi):
    from experiments.chunk_depth_v3.phase0_online import Ridge
    x = torch.tensor([phi], dtype=torch.float64)
    a = torch.cat([(x - sched["mu"]) / sched["sd"], torch.ones(1, 1, dtype=torch.float64)], 1)
    return int((a @ sched["w"])[0].add(sched["nu"] * R.R_REL).argmin())


def arm_main(arm: str, seed: int, epochs: int = 2, tag: str = "pilot"):
    torch.backends.cuda.matmul.allow_tf32 = False
    kappa_s, kappa_e = load_kv()
    items = torch.load(OUT / "items_pilot_train.pt")
    feats = torch.load(OUT / "features_pilot_train.pt")
    blob = torch.load(OUT / "readouts.pt")
    lam = blob["selected_lambda"]
    readout = new_readout()
    readout.load_state_dict(blob["readouts"][lam])
    readout = readout.cuda()
    for p in readout.parameters():
        p.requires_grad_(False)
    sched = torch.load(OUT / "sched_a_schedule.pt")
    model = H.load_model()
    # fixed reference policy from zero LoRA, per (document, line, previous depth)
    reps0 = frozen_boundary_reps(model, items, feats)
    structs = [doc_structure(it) for it in items]
    with torch.no_grad():
        mu_ref = [mean_policy(readout(torch.cat([z.cuda(), torch.tensor([l["phi"] for l in st], device="cuda")
                                                 .unsqueeze(0).expand(K, -1, -1)], -1))).cpu()
                  for z, st in zip(reps0, structs)]                   # [K(prev depth), lines, K]
    modules = H.attach_lora(model, 8, 16, seed=seed)
    params = H.lora_parameters(modules)
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    n_docs, batch = len(items), 8
    total = n_docs * epochs // batch
    order = torch.cat([torch.randperm(n_docs, generator=gen) for _ in range(epochs)])
    log, t_start = [], time.perf_counter()
    for update in range(total):
        for g_ in opt.param_groups:
            g_["lr"] = lr_at(update, total, 1e-4, 1e-5, 0.05)
        opt.zero_grad(set_to_none=True)
        stats = {"objective": 0.0, "gen_nll": 0.0, "share": torch.zeros(K)}
        for k_ in range(batch):
            j = int(order[update * batch + k_])
            item, f, st = items[j], feats[j], structs[j]
            S = len(item["ids"])
            ids = torch.tensor(item["ids"], device="cuda")
            fc = H.freqs(model, torch.arange(S))
            zs, nll = [], []
            for k in range(K):
                z = H.coda(model, f["feats"][k].cuda()[None], fc)[0]
                zs.append(z)
                nll.append(torch.nn.functional.cross_entropy(H.head(model, z[None])[0, :-1], ids[1:], reduction="none"))
            lines = [l for l in st if l["n"] > 0]
            N = sum(l["n"] for l in lines)
            L = [torch.stack([nll[k][l["scored"]].sum() for k in range(K)]) for l in lines]
            # sample the history from the current policy without gradient
            hist, kprev, sampled, mu_hist, x_idx = [], R.V3.index(16), [], [], []
            for i, l in enumerate(st):
                x = torch.cat([zs[kprev][l["a"] - 1].detach().float(), torch.tensor(l["phi"], device="cuda")])
                with torch.no_grad():
                    mu = mean_policy(readout(x))
                g2 = torch.Generator().manual_seed(seed * 10**7 + update * 1000 + k_ * 50 + i)
                k = int(torch.multinomial(mu.cpu().float(), 1, generator=g2))
                if l["n"] > 0:
                    sv = [R.coverage(hist, d) for d in R.V3]
                    mu_hist.append((mu, kprev, i, sv))
                sampled.append(k)
                hist.append((l["b"] - l["a"], R.V3[k]))
                kprev = k
            loss_gen = 0.0
            loss_route = 0.0
            for (mu0, kp, i, sv), l, Lc in zip(mu_hist, lines, L):
                cost = Lc.detach().double() + lam * l["n"] * R.R_REL.cuda() + \
                    l["n"] * torch.tensor([kappa_s * s + kappa_e * e for s, e in sv], dtype=torch.float64, device="cuda")
                if arm in ("E", "J"):
                    loss_gen = loss_gen + (mu0.detach().float() * Lc).sum()
                    x = torch.cat([zs[kp][l["a"] - 1].float(), torch.tensor(l["phi"], device="cuda")])
                    logit = readout(x)
                    alpha = alpha_of(logit)
                    mu = alpha / alpha.sum()
                    aux = aux_loss(mu, cost / l["n"], mu_ref[j][kp, i].cuda())
                    route = (mu * cost).sum() if arm == "E" else \
                        route_jev(alpha, cost, seed=seed * 10**7 + update * 1000 + k_ * 50 + i)
                    loss_route = loss_route + route + l["n"] * aux
                    with torch.no_grad():
                        stats["objective"] += float((mu * cost).sum() + l["n"] * aux) / N / batch
                elif arm == "SCHED-A":
                    loss_gen = loss_gen + Lc[sched_depth(sched, l["phi"])]
                else:
                    loss_gen = loss_gen + Lc.mean()
            loss = (loss_gen + loss_route) / N / batch
            loss.backward()
            stats["gen_nll"] += float(sum(Lc.detach()[2] for Lc in L)) / N / batch
            for k in sampled:
                stats["share"][k] += 1.0 / len(sampled) / batch
        grad_norm = float(torch.nn.utils.clip_grad_norm_(params, 1.0))
        opt.step()
        entry = {"update": update, "lr": opt.param_groups[0]["lr"], "grad_norm": grad_norm,
                 "objective_per_token": stats["objective"], "depth8_nll": stats["gen_nll"],
                 "sampled_depth_share": [round(float(v), 3) for v in stats["share"]]}
        log.append(entry)
        if update % 10 == 0 or update == total - 1:
            print(json.dumps(entry), flush=True)
    seconds = time.perf_counter() - t_start
    path = OUT / f"{tag}_{arm}_s{seed}.pt"
    torch.save({"lora": [p.detach().cpu() for p in params], "arm": arm, "seed": seed, "lambda": lam}, path)
    write_json(RESULTS / f"{tag}_train_{arm}_s{seed}.json", receipt(
        "chunk_depth_v3_1_arm_training", arm=arm, seed=seed, updates=total, epochs=epochs, lambda_=lam,
        kappa_s=kappa_s, kappa_e=kappa_e, objective="local surrogate (plan 4.3)", checkpoint=str(path),
        checkpoint_sha256=sha256_file(path), seconds=seconds, seconds_per_update=seconds / total, log=log))
    print(json.dumps({"arm": arm, "updates": total, "seconds_per_update": seconds / total}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("warm")
    a = sub.add_parser("arm")
    a.add_argument("--arm", choices=("E", "J", "SCHED-A", "FMIX-A"), required=True)
    a.add_argument("--seed", type=int, default=17)
    args = parser.parse_args()
    if args.cmd == "warm":
        warm_main()
    else:
        arm_main(args.arm, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
