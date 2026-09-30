"""
CADE for multiple-choice questions.

The candidates are the answer letters that have a logit column (logit_A, logit_B, ...).
A question's valid options are the letters whose option-text column (A, B, ...) is
non-empty in the Full-view table; without option-text columns, the letters whose Full-view
logit is present. The baseline is the model's generated answer (`prediction` in the
Full-view table); an answer that is not a valid option letter counts as wrong.
Hyperparameters maximise accuracy on the val split.

    # tables mcq_T1, mcq_T1_text, mcq_T5, mcq_T5_text in preds/<model> (SDGBiasBench T1, or any dataset)
    python -m cade.run_mcq search --pred-dir preds/<model> --out-dir runs/<model>/mcq_T1

    # SDGBiasBench context test T3
    python -m cade.run_mcq search --pred-dir preds/<model> --level 3 --out-dir runs/<model>/mcq_T3

    # or give the four tables explicitly
    python -m cade.run_mcq search --full full.csv --img img.csv --ctx ctx.csv --q q.csv --out-dir runs/my_task

    # re-apply tuned hyperparameters
    python -m cade.run_mcq apply --full full.csv --img img.csv --ctx ctx.csv --q q.csv --params runs/my_task/cade_config.json
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from .answers import parse_letter
from .cli import build_parser, run
from .core import MCQ_RANGES, accuracy_metric


def make_targets(args, base: pd.DataFrame, logits, candidates):
    if all(c in base.columns for c in candidates):
        mask = np.stack([base[c].fillna("").astype(str).str.strip().ne("").to_numpy() for c in candidates], 1)
    else:
        mask = ~torch.isnan(logits["full"]).numpy()
    letter = base["answer"].fillna("").astype(str).str.strip().str.upper()
    labels = letter.map({c: i for i, c in enumerate(candidates)}).fillna(-1).astype(int).to_numpy()
    labels[(labels >= 0) & ~mask[np.arange(len(base)), labels.clip(0)]] = -1   # gold must be a valid option
    if "prediction" not in base.columns:
        raise SystemExit("mcq: the Full-view table needs a `prediction` column (the generated answer)")
    index = {c: i for i, c in enumerate(candidates)}
    answered = [index.get(parse_letter(p, {c for c, ok in zip(candidates, row) if ok}), -1)
                for p, row in zip(base["prediction"].fillna(""), mask)]
    return {"mask": mask, "labels": labels, "answered": torch.tensor(answered), "base": base,
            "candidates": candidates}


def report(pv, targets, pred, rows):
    ok = pv.labels[rows] >= 0
    answered = targets["answered"].to(rows.device)          # generated answer; -1 = not a valid option
    p, b, y = pred[rows][ok], answered[rows][ok], pv.labels[rows][ok]
    acc = lambda x, m=slice(None): (x[m] == y[m]).double().mean().item()
    out = {"n": int(ok.sum()), "baseline_acc": acc(b), "cade_acc": acc(p),
           "changed_rate": (p != b).double().mean().item()}
    out["delta"] = out["cade_acc"] - out["baseline_acc"]
    sub = targets["base"].iloc[rows.cpu().numpy()][ok.cpu().numpy()]
    if "pillar" in sub:
        out["by_pillar"] = {}
        for val in sorted(sub["pillar"].dropna().unique()):
            m = torch.from_numpy(sub["pillar"].to_numpy() == val).to(p.device)
            out["by_pillar"][val] = {"n": int(m.sum()), "baseline_acc": acc(b, m), "cade_acc": acc(p, m)}
    return out


def fmt(m):
    return (f"n={m['n']:>6}  accuracy baseline={m['baseline_acc']:.4f}  CADE={m['cade_acc']:.4f}  "
            f"delta={m['delta']:+.4f}  changed={m['changed_rate']:.1%}")


def pred_columns(base, targets, pv, pred):
    out = base[[c for c in ("index", "split", "answer") if c in base]].copy()
    letters = np.array(list(targets["candidates"]) + [""])      # -1 -> "" (not a valid option)
    out["baseline_pred"] = letters[targets["answered"].numpy()]
    out["cade_pred"] = letters[pred.numpy()]
    out["baseline_conf"] = pv.base_conf.cpu().numpy()
    out["kl"] = pv.kl.cpu().numpy()
    return out


def main(argv=None):
    args = build_parser("python -m cade.run_mcq", "mcq", MCQ_RANGES, __doc__).parse_args(argv)
    return run(args, "mcq", MCQ_RANGES, make_targets, lambda pv, t: accuracy_metric(pv), report, fmt, pred_columns)


if __name__ == "__main__":
    main()
