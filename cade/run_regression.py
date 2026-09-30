"""
CADE for numeric answers.

CADE is applied to the first generated token (paper Suppl. B.2): the candidates are the
digits 0-9 (columns logit_0 ... logit_9). The Full-view table holds the generated answer
(`prediction`) and the gold value (`answer`). The baseline is the number in the generated
answer (0.0 when there is none); CADE's answer is that number with its first digit set to the
digit CADE selects. Hyperparameters minimise MAE on the val split.
Interval accuracy uses --tolerance, or for SDGBiasBench the per-`target` tolerance.

    # any dataset: tables regression_T1, regression_T1_text, regression_T5, regression_T5_text
    python -m cade.run_regression search --pred-dir preds/<model> --tolerance 1.0

    # SDGBiasBench context test T2
    python -m cade.run_regression search --pred-dir preds/<model> --level 2 --out-dir runs/<model>/regression_T2
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
import torch

from .answers import NUMBER, parse_number
from .cli import build_parser, run
from .core import REGRESSION_RANGES

DIGITS = tuple("0123456789")
# SDGBiasBench interval-accuracy tolerance per `target`: correct if |y_hat - y| <= tolerance.
TOLERANCE = {"under5_mort": 5.0, "women_bmi": 1.0, "asset_index": 0.75,
             "sanitation_index": 0.75, "water_index": 0.75, "women_edu": 1.5}


def digit_alternatives(prediction: str):
    """Value of the generated number if its first digit were d, for d = 0..9
    (0.0 for every d when the answer contains no number)."""
    m = NUMBER.search(str(prediction))
    if m is None:
        return [0.0] * 10
    num = m.group(0)
    k = re.search(r"\d", num).start()
    return [float(num[:k] + d + num[k + 1:]) for d in DIGITS]


def make_targets(args, base: pd.DataFrame, logits, candidates):
    if "prediction" not in base.columns:
        raise SystemExit("regression: the Full-view table needs a `prediction` column (the generated answer)")
    values = [digit_alternatives(p) for p in base["prediction"].fillna("")]
    y = pd.to_numeric(base["answer"], errors="coerce").to_numpy(np.float64)
    if args.tolerance is not None:
        tol = np.full(len(base), args.tolerance)
    elif "target" in base.columns:
        tol = base["target"].map(TOLERANCE).to_numpy(np.float64)
    else:
        tol = np.full(len(base), np.nan)
    answered = [parse_number(p) for p in base["prediction"].fillna("")]
    return {"values": torch.tensor(values, dtype=torch.float64), "answered": torch.tensor(answered, dtype=torch.float64),
            "y": torch.from_numpy(y), "tol": torch.from_numpy(tol), "base": base}


def _to(targets, device):
    for k in ("values", "answered", "y", "tol"):
        targets[k] = targets[k].to(device)
    return targets


def value_of(targets, pred, rows):
    """Numeric answers for (K, n) candidate indices on `rows`."""
    vals = targets["values"][rows]
    return vals[None].expand(pred.shape[0], -1, -1).gather(-1, pred[..., None]).squeeze(-1)


def neg_mae_metric(pv, targets):
    _to(targets, pv.mask.device)

    def metric(pred, rows):
        y = targets["y"][rows]
        ok = ~torch.isnan(y)
        return -(value_of(targets, pred, rows) - y).abs()[:, ok].mean(-1)
    return metric


def report(pv, targets, pred, rows):
    _to(targets, pv.mask.device)
    y, tol = targets["y"][rows], targets["tol"][rows]
    ok = ~torch.isnan(y)
    has_tol = ok & ~torch.isnan(tol)
    baseline = targets["answered"][rows]                    # number in the generated answer
    cade = value_of(targets, pred[rows][None], rows)[0]
    out = {"n": int(ok.sum()), "changed_rate": (cade != baseline)[ok].double().mean().item()}
    for name, v in (("baseline", baseline), ("cade", cade)):
        err = (v - y).abs()
        out[f"{name}_mae"] = err[ok].mean().item()
        if has_tol.any():
            out[f"{name}_interval_acc"] = (err[has_tol] <= tol[has_tol]).double().mean().item()
    out["delta_mae"] = out["cade_mae"] - out["baseline_mae"]
    return out


def fmt(m):
    s = f"n={m['n']:>6}  MAE baseline={m['baseline_mae']:.3f}  CADE={m['cade_mae']:.3f} ({m['delta_mae']:+.3f})"
    if "cade_interval_acc" in m:
        s += f"  interval_acc {m['baseline_interval_acc']:.4f} -> {m['cade_interval_acc']:.4f}"
    return s + f"  changed={m['changed_rate']:.1%}"


def pred_columns(base, targets, pv, pred):
    out = base[[c for c in ("index", "split", "answer") if c in base]].copy()
    rows = torch.arange(len(base), device=pv.mask.device)
    out["baseline_pred"] = targets["answered"].cpu().numpy()
    out["cade_pred"] = value_of(targets, pred.to(rows.device)[None], rows)[0].cpu().numpy()
    out["baseline_conf"] = pv.base_conf.cpu().numpy()
    out["kl"] = pv.kl.cpu().numpy()
    return out


def main(argv=None):
    args = build_parser("python -m cade.run_regression", "regression", REGRESSION_RANGES, __doc__).parse_args(argv)
    return run(args, "regression", REGRESSION_RANGES, make_targets, neg_mae_metric, report, fmt, pred_columns,
               candidates=DIGITS)


if __name__ == "__main__":
    main()
