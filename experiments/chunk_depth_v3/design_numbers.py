"""Numbers needed by the v3 design: split sizes and per-item effect spread for sample sizes."""

from __future__ import annotations

import json
import re

import pyarrow.parquet as pq
import torch
from transformers import AutoTokenizer

from experiments.shared_depth_v2.common import MODEL_DIR, receipt, write_json
from experiments.chunk_depth_v3.analyze_survey import static_best
from experiments.chunk_depth_v3.probe import CHOICES, FOLDS, crossfit, decision_units, position_features
from experiments.chunk_depth_v3.survey import BOS, DATA, OUT, RESULTS, bucket


def lengths(split: str, tok) -> list[tuple[int, int, int]]:
    rows = pq.read_table(DATA / f"{split}.parquet").to_pylist()
    out = []
    for i, r in enumerate(rows):
        lines = re.sub(r"<<[^>]*>>", "", r["answer"]).strip().split("\n")
        sol = "\n".join(l for l in lines if not l.startswith("####"))
        text = f"Question: {r['question'].strip()}\nAnswer: {sol}"
        n = 1 + len(tok(text, add_special_tokens=False, split_special_tokens=True)["input_ids"])
        out.append((i, bucket(r["question"]), n))
    return out


def main() -> int:
    tok = AutoTokenizer.from_pretrained(str(MODEL_DIR))
    train, test = lengths("train", tok), lengths("test", tok)
    ranges = {"survey": (0, 9), "formal_train": (10, 64), "warm": (65, 69), "pilot_train": (70, 74),
              "pilot_cal": (75, 77), "pilot_screen": (78, 84), "formal_cal": (85, 89), "formal_screen": (90, 99)}
    counts = {name: sum(1 for _, b, n in train if lo <= b <= hi and n <= 320) for name, (lo, hi) in ranges.items()}
    counts["confirm_test_le_320"] = sum(1 for *_, n in test if n <= 320)
    counts["train_total"], counts["test_total"] = len(train), len(test)
    tl = torch.tensor([n for *_, n in train], dtype=torch.float64)
    blob = torch.load(OUT / "survey_tables.pt")
    items, tables = blob["items"], blob["tables"]
    units = decision_units(items, tables)
    reps = torch.load(OUT / "boundary_reps_depth8.pt")
    y = torch.stack([u["nll"] for u in units])
    cost = torch.stack([u["cost"] for u in units])
    base16 = float(cost[:, CHOICES.index(16)].sum())
    item = torch.tensor([u["item"] for u in units])
    ntok = torch.tensor([len(u["scored"]) for u in units], dtype=torch.float64)
    spread = {}
    for B in (0.4, 0.5, 0.6):
        budget = lambda c: B * float(c[:, CHOICES.index(16)].sum())
        sched = crossfit(position_features(units), units, FOLDS, 10.0, budget)
        rep = crossfit(torch.cat([reps.double(), position_features(units) * 30.0], 1), units, FOLDS, 10000.0, budget)
        st = static_best(y.sum(0), cost.sum(0), B * base16)
        st_unit = sum(w * y[:, CHOICES.index(d)] for d, w in st["support"])
        idx = torch.arange(len(units))
        n_items = int(item.max()) + 1

        def per_item(v):
            num = torch.zeros(n_items, dtype=torch.float64).index_add_(0, item, v)
            den = torch.zeros(n_items, dtype=torch.float64).index_add_(0, item, ntok)
            return num / den

        d_static_sched = per_item(st_unit - y[idx, sched])
        d_sched_rep = per_item(y[idx, sched] - y[idx, rep])
        spread[B] = {"schedule_vs_static_mean": float(d_static_sched.mean()), "schedule_vs_static_sd": float(d_static_sched.std()),
                     "rep_vs_schedule_mean": float(d_sched_rep.mean()), "rep_vs_schedule_sd": float(d_sched_rep.std()),
                     "items": n_items}
    payload = receipt("chunk_depth_v3_design_numbers", counts=counts,
                      train_tokens={"mean": float(tl.mean()), "q95": float(torch.quantile(tl, 0.95)),
                                    "share_le_320": float((tl <= 320).double().mean())},
                      per_item_effect_spread=spread)
    write_json(RESULTS / "design_numbers.json", payload)
    print(json.dumps({"counts": counts, "spread": spread}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
