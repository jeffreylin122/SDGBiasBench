#!/usr/bin/env bash
# Run CADE (search on val, report on test) for every context test of every model.
#   bash scripts/run_cade_all.sh preds runs Qwen2.5-VL-7B-Instruct internvl3_8b ...
# expects PRED_ROOT/<model>/ to hold the per-view prediction files (see README).
set -euo pipefail
PRED_ROOT=${1:?usage: run_cade_all.sh PRED_ROOT OUT_ROOT MODEL...}
OUT_ROOT=${2:?usage: run_cade_all.sh PRED_ROOT OUT_ROOT MODEL...}
shift 2
for model in "$@"; do
  for k in 1 2 3 4 5; do
    python -m cade.run_mcq search --pred-dir "$PRED_ROOT/$model" --level "$k" \
      --out-dir "$OUT_ROOT/$model/mcq_T$k" --save-preds
  done
  for k in 2 5; do
    python -m cade.run_regression search --pred-dir "$PRED_ROOT/$model" --level "$k" \
      --out-dir "$OUT_ROOT/$model/regression_T$k" --save-preds
  done
done
