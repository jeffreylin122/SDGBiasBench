"""
CADE core: Contrastive Adaptive Debias Ensemble (paper Sec. 4, Eqs. 1-10).

Everything is vectorised over rows *and* hyperparameter configurations, so a random
search of 10k configurations over ~100k questions runs on one GPU in seconds.

Notation (per datapoint i, candidate a in O_i):
    p_full, p_img, p_ctx, p_q  per-view probabilities            (Eq. 1)
    a_base = argmax p_full,  m = max p_full                        (Eq. 2)
    p_img_stream = norm(p_full + p_img), p_ctx_stream = p_ctx,
    p_bias_stream = p_q                                            (Eqs. 3-6)
    D     = KL(p_img_stream || p_ctx_stream)                       (Eq. 7)
    alpha_i = alpha * (1 + lambda_kl * D)                          (Eq. 8)
    score = log p_img - alpha_i log p_ctx - beta log p_q           (Eq. 9)
    pred  = a_base if m >= tau else argmax score                   (Eqs. 2, 10)

MCQ: O_i = answer letters. Regression (Suppl. B.2): O_i = first-token digits {0..9}.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Callable, Dict, Optional, Tuple

import torch

EPS = 1e-9
VIEWS = ("full", "img", "ctx", "q")
PARAM_NAMES = ("alpha", "lambda_kl", "beta", "tau")

# Random-search ranges (paper Table G).
MCQ_RANGES: Dict[str, Tuple[float, float]] = {
    "alpha": (0.5, 2.0), "lambda_kl": (0.0, 5.0), "beta": (0.0, 1.5), "tau": (0.65, 0.99)}
REGRESSION_RANGES: Dict[str, Tuple[float, float]] = {
    "alpha": (0.0, 7.0), "lambda_kl": (0.0, 7.0), "beta": (0.0, 7.0), "tau": (0.25, 0.99)}

# metric(pred, rows) -> (K,) score, higher is better. `pred` is (K, len(rows)) candidate
# indices for the rows `rows` of the PreparedViews the search runs on.
Metric = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class CADEParams:
    """One CADE configuration. Defaults (tau=0: the gate always keeps the baseline)
    reproduce the plain Full-view prediction."""
    alpha: float = 0.0
    lambda_kl: float = 0.0
    beta: float = 0.0
    tau: float = 0.0  # keep the baseline iff max p_full >= tau

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)


@dataclass
class PreparedViews:
    """Per-row quantities that do not depend on hyperparameters (computed once)."""
    log_img: torch.Tensor    # (N, O) log of image stream        (Eqs. 3, 6)
    log_ctx: torch.Tensor    # (N, O) log of context stream      (Eqs. 4, 6)
    log_bias: torch.Tensor   # (N, O) log of bias stream         (Eqs. 5, 6)
    kl: torch.Tensor         # (N,)   D_i                        (Eq. 7)
    base_pred: torch.Tensor  # (N,)   argmax p_full               (Eq. 2)
    base_conf: torch.Tensor  # (N,)   max p_full                 (Eq. 2)
    mask: torch.Tensor       # (N, O) True for valid candidates of row i
    labels: Optional[torch.Tensor] = None  # (N,) gold candidate index, -1 if unknown
    missing: Dict[str, int] = field(default_factory=dict)  # rows lacking each view

    def __len__(self) -> int:
        return self.mask.shape[0]

    def subset(self, idx: torch.Tensor) -> "PreparedViews":
        pick = lambda t: None if t is None else t[idx]
        return PreparedViews(
            pick(self.log_img), pick(self.log_ctx), pick(self.log_bias), pick(self.kl),
            pick(self.base_pred), pick(self.base_conf), pick(self.mask), pick(self.labels),
            dict(self.missing),
        )


def _normalize(p: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Renormalise over valid candidates; rows with no mass become uniform."""
    p = p.masked_fill(~mask, 0.0)
    s = p.sum(-1, keepdim=True)
    uniform = mask / mask.sum(-1, keepdim=True)
    return torch.where(s > 0, p / s.clamp_min(EPS), uniform)


def _to_probs(z: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Eq. 1: softmax of the logits over the valid candidates. Returns (probabilities, present);
    present[i] is False when the view has no logits for row i (NaN on every valid candidate)."""
    present = (~torch.isnan(z) & mask).any(-1)
    p = torch.nan_to_num(z, nan=0.0).masked_fill(~mask, float("-inf")).softmax(-1)
    return torch.where(present[:, None], p, mask / mask.sum(-1, keepdim=True)), present


def prepare_views(
    views: Dict[str, torch.Tensor],
    mask: Optional[torch.Tensor] = None,
    labels: Optional[torch.Tensor] = None,
) -> PreparedViews:
    """Build the three streams (Eqs. 1-7) from the four views' logits.

    Args:
        views: {"full","img","ctx","q"} -> (N, O) first-token logits of the candidates.
            NaN marks a view without logits for that row; it is treated as uniform (no
            evidence), and the image stream sums only the available views among Full / IMG+Q.
        mask:  (N, O) bool of valid candidates (e.g. 2- vs 3-option MCQs); default all.
        labels: optional (N,) gold candidate indices (-1 = unknown).
    """
    if set(views) != set(VIEWS):
        raise ValueError(f"views must have exactly the keys {VIEWS}, got {sorted(views)}")
    ref = views["full"]
    mask = torch.ones(ref.shape, dtype=torch.bool, device=ref.device) if mask is None else mask.bool()
    probs, present = {}, {}
    for v in VIEWS:
        probs[v], present[v] = _to_probs(views[v].to(mask.device, torch.float64), mask)

    # Eq. 2: the Full view is the baseline.
    base_conf, base_pred = probs["full"].masked_fill(~mask, -1.0).max(-1)

    # Eqs. 3-6: three streams. Image stream sums whichever of Full / IMG+Q exist.
    img_sum = probs["full"] * present["full"][:, None] + probs["img"] * present["img"][:, None]
    p_img = _normalize(img_sum, mask)
    p_ctx, p_bias = probs["ctx"], probs["q"]

    log = lambda p: torch.log(p.clamp_min(EPS)).masked_fill(~mask, 0.0)
    log_img, log_ctx, log_bias = log(p_img), log(p_ctx), log(p_bias)

    # Eq. 7: D_i = KL(p_img || p_ctx), over valid candidates only.
    kl = (p_img * (log_img - log_ctx)).masked_fill(~mask, 0.0).sum(-1).clamp_min(0.0)

    return PreparedViews(
        log_img, log_ctx, log_bias, kl, base_pred, base_conf, mask,
        None if labels is None else labels.to(mask.device).long(),
        {v: int((~present[v]).sum()) for v in VIEWS},
    )


def _as_param_tensors(params, device) -> Dict[str, torch.Tensor]:
    """CADEParams | list of CADEParams | dict of 1-D tensors -> dict of (K,) tensors."""
    if isinstance(params, CADEParams):
        params = [params]
    if isinstance(params, (list, tuple)):
        params = {k: torch.tensor([getattr(p, k) for p in params]) for k in PARAM_NAMES}
    return {k: torch.as_tensor(params[k], dtype=torch.float64, device=device) for k in PARAM_NAMES}


def predict(pv: PreparedViews, params) -> torch.Tensor:
    """Eqs. 8-10 for K configurations at once. Returns (K, N) candidate indices."""
    P = _as_param_tensors(params, pv.mask.device)
    alpha_i = P["alpha"][:, None] * (1.0 + P["lambda_kl"][:, None] * pv.kl[None])          # Eq. 8
    score = (pv.log_img[None]
             - alpha_i[..., None] * pv.log_ctx[None]
             - P["beta"][:, None, None] * pv.log_bias[None])                                # Eq. 9
    debiased = score.masked_fill(~pv.mask[None], float("-inf")).argmax(-1)                 # Eq. 10
    keep_base = pv.base_conf[None] >= P["tau"][:, None]                                    # Eq. 2 gate
    return torch.where(keep_base, pv.base_pred[None], debiased)


def accuracy_metric(pv: PreparedViews) -> Metric:
    """Default MCQ objective: accuracy against pv.labels (rows without a label are ignored)."""
    def metric(pred, rows):
        y = pv.labels[rows]
        return (pred == y[None])[:, y >= 0].double().mean(-1)
    return metric


def evaluate_configs(pv: PreparedViews, params, metric: Metric, rows: torch.Tensor,
                     chunk: int = 512) -> torch.Tensor:
    """Score K configurations on the given rows of `pv` -> (K,)."""
    P = _as_param_tensors(params, pv.mask.device)
    sub = pv.subset(rows)
    K = P["alpha"].numel()
    return torch.cat([metric(predict(sub, {k: v[s:s + chunk] for k, v in P.items()}), rows)
                      for s in range(0, K, chunk)])


def sample_params(
    n: int,
    ranges: Dict[str, Tuple[float, float]] = MCQ_RANGES,
    generator: Optional[torch.Generator] = None,
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    """Draw n configurations uniformly in `ranges`."""
    out = {}
    for k in PARAM_NAMES:
        lo, hi = ranges[k]
        u = torch.rand(n, generator=generator, dtype=torch.float64)
        out[k] = (lo + (hi - lo) * u).to(device)
    return out


@dataclass
class SearchResult:
    best: CADEParams
    search_score: float     # metric of `best` on the search subset
    val_score: float        # metric of `best` on the full validation rows
    n_trials: int
    n_search_rows: int
    n_val_rows: int


def random_search(
    pv: PreparedViews,
    val_rows: torch.Tensor,
    metric: Optional[Metric] = None,
    n_trials: int = 10_000,
    search_frac: float = 0.2,
    top_k: int = 100,
    ranges: Dict[str, Tuple[float, float]] = MCQ_RANGES,
    seed: int = 0,
    chunk: int = 512,
) -> SearchResult:
    """Random search of Suppl. B.4: sample `n_trials` configurations uniformly in `ranges`,
    score each once on a random `search_frac` of the validation rows, re-score the `top_k`
    best on all validation rows and return the winner. `metric` defaults to accuracy."""
    device = pv.mask.device
    metric = metric or accuracy_metric(pv)
    val_rows = val_rows.to(device)
    gen = torch.Generator().manual_seed(seed)
    cand = sample_params(n_trials, ranges, gen, device)

    n_search = max(1, math.ceil(search_frac * val_rows.numel()))
    search_rows = val_rows[torch.randperm(val_rows.numel(), generator=gen).to(device)[:n_search]]

    s_search = evaluate_configs(pv, cand, metric, search_rows, chunk)
    top = s_search.topk(min(top_k, n_trials)).indices
    s_val = evaluate_configs(pv, {k: v[top] for k, v in cand.items()}, metric, val_rows, chunk)
    j = int(s_val.argmax())
    best_idx = int(top[j])
    return SearchResult(
        best=CADEParams(**{k: float(cand[k][best_idx]) for k in PARAM_NAMES}),
        search_score=float(s_search[best_idx]),
        val_score=float(s_val[j]),
        n_trials=n_trials,
        n_search_rows=n_search,
        n_val_rows=int(val_rows.numel()),
    )
