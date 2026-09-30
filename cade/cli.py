"""Command-line plumbing shared by `cade.run_mcq` and `cade.run_regression`."""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Callable, Dict

import numpy as np
import torch

from .core import PARAM_NAMES, VIEWS, CADEParams, predict, prepare_views, random_search
from .io import load_views

# View tables in --pred-dir for context test T<k> (T5 carries no context, so there Full = IMG+Q
# and CTX+Q = Q-only). A dataset without context tests names its tables as T1.
VIEW_FILES = {"full": "{task}_T{k}", "img": "{task}_T5", "ctx": "{task}_T{k}_text", "q": "{task}_T5_text"}


def build_parser(prog: str, task: str, ranges, doc: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog=prog, description=doc, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    g = common.add_argument_group("input: one prediction table per view")
    for v in VIEWS:
        g.add_argument(f"--{v}", metavar="FILE", help=f"{v} view table")
    g.add_argument("--pred-dir", help=f"folder with the view tables {task}_T<k>, {task}_T<k>_text, {task}_T5, {task}_T5_text")
    g.add_argument("--level", type=int, default=1, help="context test k in the table names (SDGBiasBench: 1-5 for MCQ, 2 or 5 for regression)")
    if task == "regression":
        g.add_argument("--tolerance", type=float,
                       help="interval-accuracy tolerance (default: per SDGBiasBench `target`)")
    g = common.add_argument_group("output / runtime")
    g.add_argument("--out-dir", help="write cade_config.json (and predictions with --save-preds) here")
    g.add_argument("--save-preds", action="store_true", help="also write cade_predictions.csv")
    g.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    s = sub.add_parser("search", parents=[common], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                       help="tune the hyperparameters on the val split, report on the test split")
    g = s.add_argument_group("random search")
    g.add_argument("--n-trials", type=int, default=10_000)
    g.add_argument("--search-frac", type=float, default=0.2, help="fraction of val each trial is scored on")
    g.add_argument("--top-k", type=int, default=100, help="configs re-scored on the full val split")
    for k, (lo, hi) in ranges.items():
        g.add_argument(f"--{k.replace('_', '-')}-range", type=float, nargs=2, default=[lo, hi], metavar=("LO", "HI"))
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--chunk", type=int, default=512, help="configs evaluated per GPU batch")

    a = sub.add_parser("apply", parents=[common], formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                       help="apply given hyperparameters and report")
    g = a.add_argument_group("hyperparameters (--params and/or explicit values)")
    g.add_argument("--params", help="cade_config.json from a previous `search` run")
    for k in PARAM_NAMES:
        g.add_argument(f"--{k.replace('_', '-')}", type=float)
    return p


def resolve_paths(args, task: str) -> Dict[str, str]:
    paths = {}
    if args.pred_dir:
        for v, pattern in VIEW_FILES.items():
            stem = os.path.join(args.pred_dir, pattern.format(task=task, k=args.level))
            found = [stem + ext for ext in (".csv", ".parquet", ".xlsx") if os.path.exists(stem + ext)]
            if found:
                paths[v] = found[0]
    paths.update({v: getattr(args, v) for v in VIEWS if getattr(args, v)})
    missing = [v for v in VIEWS if v not in paths]
    if missing:
        where = (f" (looked for {', '.join(VIEW_FILES[v].format(task=task, k=args.level) for v in missing)}"
                 f" in {args.pred_dir}; set --level?)" if args.pred_dir else "")
        raise SystemExit(f"no table for view(s) {missing}{where}: pass --pred-dir, or --{missing[0]} FILE")
    return paths


def params_from_args(args) -> CADEParams:
    values = {}
    if args.params:
        with open(args.params) as f:
            values = json.load(f)["params"]
    values.update({k: getattr(args, k) for k in PARAM_NAMES if getattr(args, k) is not None})
    missing = [k for k in PARAM_NAMES if k not in values]
    if missing:
        raise SystemExit(f"apply: missing hyperparameters {missing} (use --params or --{missing[0].replace('_', '-')})")
    return CADEParams(**{k: float(values[k]) for k in PARAM_NAMES})


def run(args, task: str, ranges, make_targets: Callable, metric_fn: Callable,
        report_fn: Callable, fmt_fn: Callable, pred_columns: Callable, candidates=None):
    """Load views -> prepare -> (search) -> report -> save. Task-specific callbacks:
        make_targets(args, base, logits, candidates) -> dict with optional mask / labels
        metric_fn(pv, targets) -> objective for random_search (higher is better)
        report_fn(pv, targets, pred, rows) -> metrics dict for the given rows
        fmt_fn(metrics) -> one-line summary
        pred_columns(base, targets, pv, pred) -> per-question output table
    """
    device = torch.device(args.device)
    paths = resolve_paths(args, task)
    t0 = time.time()
    base, logits, candidates = load_views(paths, task, candidates)
    targets = make_targets(args, base, logits, candidates)
    as_t = lambda x: None if x is None else torch.as_tensor(x).to(device)
    pv = prepare_views({k: v.to(device) for k, v in logits.items()}, as_t(targets.get("mask")),
                       as_t(targets.get("labels")))
    split = base["split"].to_numpy()
    rows = {s: torch.from_numpy(np.flatnonzero(split == s)).to(device) for s in ("val", "test")}
    print(f"[data] {len(base)} questions (val={len(rows['val'])}, test={len(rows['test'])}), "
          f"candidates={candidates}, device={device}, {time.time() - t0:.1f}s")
    for v in VIEWS:
        print(f"[data]   {v:>4}: {paths[v]}")
    if any(pv.missing.values()):
        print(f"[warn] questions missing from a view (treated as no evidence): {pv.missing}")

    record = {"task": task, "views": paths, "candidates": candidates, "missing_views": pv.missing,
              "split_sizes": {s: int(len(r)) for s, r in rows.items()}}
    if args.mode == "search":
        rng = {k: tuple(getattr(args, f"{k}_range")) for k in ranges}
        t0 = time.time()
        res = random_search(pv, rows["val"], metric_fn(pv, targets), args.n_trials, args.search_frac,
                            args.top_k, rng, args.seed, args.chunk)
        params = res.best
        print(f"[search] {res.n_trials} trials on {res.n_search_rows} val questions, top-{args.top_k} "
              f"re-scored on {res.n_val_rows}, {time.time() - t0:.1f}s")
        record["search"] = {"ranges": rng, "n_trials": res.n_trials, "search_frac": args.search_frac,
                            "top_k": args.top_k, "seed": args.seed}
    else:
        params = params_from_args(args)

    print("[params] " + "  ".join(f"{k}={v:.4f}" for k, v in params.as_dict().items()))
    pred = predict(pv, params)[0]
    metrics = {s: report_fn(pv, targets, pred, rows[s]) for s in ("test", "val") if len(rows[s])}
    for s, m in metrics.items():
        print(f"[{s:<4}] {fmt_fn(m)}")
    record.update(params=params.as_dict(), metrics=metrics)

    if args.out_dir:
        os.makedirs(args.out_dir, exist_ok=True)
        path = os.path.join(args.out_dir, "cade_config.json")
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
        print(f"[out] {path}")
        if args.save_preds:
            path = os.path.join(args.out_dir, "cade_predictions.csv")
            pred_columns(base, targets, pv, pred.cpu()).to_csv(path, index=False)
            print(f"[out] {path}")
    return record
