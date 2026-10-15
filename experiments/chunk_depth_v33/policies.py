"""Decision rules of v3.3 (plan sections 4.3, 5.2, 5.3) and vectorised cost prediction on a parameter grid.

All Part-A rules have the form
    d_c = argmin_d [ ell_hat_c(d) + kappa_s * s_c(d) + kappa_e * e_c(d) + nu * r(d) ],
with s_c, e_c the shortfall and excess of the executed history (v3.1 section 2.5). R predicts ell_hat from the
position features only; H from [z; one-hot(previous depth); phi].
"""

from __future__ import annotations

import torch

from experiments.chunk_depth_v31 import run as R
from experiments.chunk_depth_v31.train import K, doc_structure, load_kv

V3_T = torch.tensor(R.V3, dtype=torch.float64)
R_REL = R.R_REL.double()


class Ridge:
    """Prediction side of a fitted ridge model (state from experiments.chunk_depth_v32.head.ridge_state)."""

    def __init__(self, state: dict):
        self.w, self.mu, self.sd = state["w"].double().cpu(), state["mu"].double().cpu(), state["sd"].double().cpu()

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        a = torch.cat([(x - self.mu) / self.sd, torch.ones(*x.shape[:-1], 1, dtype=x.dtype)], -1)
        return a @ self.w


def kappas() -> tuple[float, float]:
    return load_kv()


def kv_vector(hist, kappa) -> torch.Tensor:
    ks, ke = kappa
    return torch.tensor([ks * s + ke * e for s, e in (R.coverage(hist, d) for d in R.V3)], dtype=torch.float64)


def h_features(z: torch.Tensor, kprev: int, phi) -> torch.Tensor:
    onehot = torch.zeros(K, dtype=torch.float64)
    onehot[kprev] = 1.0
    return torch.cat([z.double(), onehot, torch.as_tensor(phi, dtype=torch.float64)])


# ----------------------------------------------------------------------------- online policies

def r_policy(ridge: Ridge, nu: float, kappa):
    def pol(z, dz, c, prev_len, hist):
        ell = ridge(torch.tensor([R.pos_feats(c, prev_len)], dtype=torch.float64))[0]
        return R.V3[int((ell + kv_vector(hist, kappa) + nu * R_REL).argmin())]
    return pol


def h_policy(ridge: Ridge, nu: float, kappa, probe=None):
    """probe: optional dict {"reps": table representations [K, lines, 5280], "agree": [n_agree, n]}; at every
    decision the rule is re-evaluated on the table representation of the same history depth."""
    def pol(z, dz, c, prev_len, hist):
        phi = R.pos_feats(c, prev_len)
        kprev = R.V3.index(dz)
        extra = kv_vector(hist, kappa) + nu * R_REL
        k = int((ridge(h_features(z, kprev, phi)[None])[0] + extra).argmin())
        if probe is not None:
            kt = int((ridge(h_features(probe["reps"][kprev, c - 1], kprev, phi)[None])[0] + extra).argmin())
            probe["agree"][0] += int(kt == k)
            probe["agree"][1] += 1
        return R.V3[k]
    return pol


def uniform_policy(d: int):
    return lambda z, dz, c, prev_len, hist: d


def fixed_policy(ridge: Ridge, nu: float = 1.47):
    """v3.2 position schedule (no KV term), the fixed schedule pi_F of experiment B."""
    def pol(z, dz, c, prev_len, hist):
        ell = ridge(torch.tensor([R.pos_feats(c, prev_len)], dtype=torch.float64))[0]
        return R.V3[int((ell + nu * R_REL).argmin())]
    return pol


# ----------------------------------------------------------------------------- vectorised cost prediction

def doc_scores_R(item, ridge: Ridge):
    st = doc_structure(item)
    ell = ridge(torch.tensor([l["phi"] for l in st], dtype=torch.float64))                # [lines, K]
    return {"ell": ell[:, None, :].expand(-1, K, -1), "m": [l["b"] - l["a"] for l in st],
            "f": torch.stack([l["f"] for l in st])}


def doc_scores_H(item, table, ridge: Ridge):
    st = doc_structure(item)
    rows = []
    for i, l in enumerate(st):
        x = torch.stack([h_features(table["reps"][k, i], k, l["phi"]) for k in range(K)])  # [K_prev, F]
        rows.append(ridge(x))
    return {"ell": torch.stack(rows), "m": [l["b"] - l["a"] for l in st], "f": torch.stack([l["f"] for l in st])}


def predicted_cost(scores: list, nus: torch.Tensor, kappa) -> torch.Tensor:
    """Relative cost of the sequential rule for every nu at once (history and KV terms follow each nu's own
    decisions; line 1 follows the depth-16 question). Returns [P]."""
    ks, ke = kappa
    P = len(nus)
    num = torch.zeros(P, dtype=torch.float64)
    den = 0.0
    for doc in scores:
        kprev = torch.full((P,), 4, dtype=torch.long)
        hist_d, hist_m = [], []
        for i in range(len(doc["m"])):
            ell = doc["ell"][i][kprev]                                                       # [P, K]
            cost = ell + nus[:, None] * R_REL[None]
            if hist_d:
                D = torch.stack(hist_d, 1)                                                   # [P, i]
                m = torch.tensor(hist_m, dtype=torch.float64)
                diff = V3_T[None, None, :] - D[:, :, None]                                   # [P, i, K]
                s = (diff.clamp(min=0) * m[None, :, None]).sum(1) / m.sum()
                e = ((-diff).clamp(min=0) * m[None, :, None]).sum(1) / m.sum()
                cost = cost + ks * s + ke * e
            k = cost.argmin(1)
            num += doc["f"][i][k]
            den += float(doc["f"][i][4])
            hist_d.append(V3_T[k])
            hist_m.append(doc["m"][i])
            kprev = k
    return num / den


def choose_parameters(grid: torch.Tensor, pred: torch.Tensor, targets) -> list[dict]:
    """For each target the grid value with the nearest predicted cost (ties to the smaller parameter)."""
    out = []
    for t in targets:
        j = int((pred - t).abs().argmin())
        out.append({"target": float(t), "param": round(float(grid[j]), 6), "predicted_cost": float(pred[j])})
    return out
