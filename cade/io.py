"""Read the four per-view prediction tables CADE needs.

Each view is one table (.csv, .parquet or .xlsx) with one row per question:
    index       question id, the same in all four views
    logit_<c>   first-token logit of every answer candidate c (e.g. logit_A ... logit_D, or logit_0 ... logit_9)
The Full-view table also holds the gold `answer` and, optionally, `split` ("val" / "test").
Without `split`, 20% of the questions (chosen by a hash of `index`) form the val split.
"""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch

VAL_FRAC = 0.2


def read_table(path: str) -> pd.DataFrame:
    """Read a prediction table; `prediction` is kept as text so its digits are not altered."""
    text = {"prediction": str}
    if path.endswith((".xlsx", ".xls")):
        return pd.read_excel(path, dtype=text)
    if path.endswith(".parquet"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype=text, encoding="utf-8-sig", low_memory=False)


def hash_split(index: pd.Series, salt: str) -> np.ndarray:
    """md5("<salt>-<index>") < VAL_FRAC -> "val", else "test" (the rule used for SDGBiasBench)."""
    u = index.map(lambda i: int(hashlib.md5(f"{salt}-{int(i)}".encode()).hexdigest()[:15], 16) / 16 ** 15)
    return np.where(u < VAL_FRAC, "val", "test")


def letter_candidates(columns) -> List[str]:
    """Answer letters that have a logit column, e.g. ['A', 'B', 'C', 'D']."""
    return sorted(c[len("logit_"):] for c in columns if re.fullmatch(r"logit_[A-Z]", c))


def _indexed(df: pd.DataFrame, path: str) -> pd.DataFrame:
    if "index" not in df.columns:
        raise SystemExit(f"{path}: needs an `index` column identifying each question")
    return df.drop_duplicates("index").set_index("index", drop=False)


def load_views(paths: Dict[str, str], salt: str, candidates: Optional[Sequence[str]] = None
               ) -> Tuple[pd.DataFrame, Dict[str, torch.Tensor], List[str]]:
    """Returns (Full-view table, {view: (N, O) float64 logits}, candidates).
    Other views are matched to the Full view by `index`; a question missing from a view
    gets NaN logits, which CADE treats as a view without evidence."""
    base = _indexed(read_table(paths["full"]), paths["full"])
    if "answer" not in base.columns:
        raise SystemExit(f"{paths['full']}: the Full-view table needs an `answer` column")
    if "split" not in base.columns:
        base["split"] = hash_split(base["index"], salt)
    candidates = list(candidates or letter_candidates(base.columns))
    if not candidates:
        raise SystemExit(f"{paths['full']}: no logit_<candidate> columns found")
    cols = [f"logit_{c}" for c in candidates]
    logits = {}
    for view, path in paths.items():
        df = base if view == "full" else _indexed(read_table(path), path)
        absent = [c for c in cols if c not in df.columns]
        if absent:
            raise SystemExit(f"{view} view ({path}): missing columns {absent}")
        values = df[cols].apply(pd.to_numeric, errors="coerce").reindex(base.index)
        logits[view] = torch.from_numpy(values.to_numpy(np.float64))
    return base.reset_index(drop=True), logits, candidates
