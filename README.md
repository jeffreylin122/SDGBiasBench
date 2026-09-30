# SDGBiasBench: Benchmarking and Mitigating Vision–Language Models' Biases in Sustainable Development Goals

[Paper](https://arxiv.org/abs/2605.21919) · [Dataset (Hugging Face)](https://huggingface.co/datasets/jeffreylin1222/SDGBiasBench)

**SDGBiasBench**

- **Motivation:** assessing progress on the Sustainable Development Goals (SDGs) requires
multi-step reasoning over visual cues, contextual knowledge and development indicators. When
a model uses this evidence incompletely or integrates it imperfectly, hidden prediction biases follow.
- **Benchmark:** SDG-oriented vision–language reasoning with both qualitative judgments and
quantitative estimation, evaluated under controlled evidence availability.
- **Finding:** current VLMs show an intrinsic **SDG bias**. They rely on SDG-specific priors
rather than multi-modal evidence, which appears as:
  - pillar-conditioned directional defaults: optimistic, conservative or pessimistic, depending on the SDG theme;
  - modality imbalance: structured context is treated as largely sufficient while imagery is under-used.

**CADE** (Contrastive Adaptive Debias Ensemble)

- Training-free and plug-and-play: works on a VLM's output logits without changing model parameters.
- Contrasts predictions from four input views (Full, image-only, context-only, question-only) to detect over-reliance on priors.
- Keeps the Full-view answer when the model is confident. Otherwise it reweights toward visual
evidence, penalising the context prior in proportion to how much the image and context views
disagree, and penalising the question-only prior.

This repository contains:

- `sdgbiasbench/` — turns the Hugging Face release into one input file per evidence view.
- `cade/` — **CADE**, usable on SDGBiasBench and on any task that provides the four views.



## Benchmark at a glance


|            | MCQ                                                           | Regression                    |
| ---------- | ------------------------------------------------------------- | ----------------------------- |
| Questions  | 500,000                                                       | 96338                         |
| Task types | 12 indicators across 3 pillars                                | 9 tasks over 6 DHS indicators |
| Metric     | accuracy                                                      | MAE, interval accuracy        |


Every question is evaluated under four **evidence views** (paper Sec. 3.1):


| view   | inputs                     | file (for context test T*k*) |
| ------ | -------------------------- | ---------------------------- |
| Full   | image + context + question | `<task>_T<k>`                |
| IMG+Q  | image + question           | `<task>_T5`                  |
| CTX+Q  | context + question         | `<task>_T<k>_text`           |
| Q-only | question                   | `<task>_T5_text`             |


See the [dataset card](https://huggingface.co/datasets/jeffreylin1222/SDGBiasBench) for fields,
splits and how the data was built.

## Repository layout

```
sdgbiasbench/
  make_views.py      Hugging Face release -> one input file per evidence view
cade/
  core.py            CADE (Eqs. 1-10) and its hyperparameter search
  io.py              reading the per-view prediction tables
  answers.py         reading the model's generated answer
  run_mcq.py         CADE command line for multiple-choice answers
  run_regression.py  CADE command line for numeric answers
scripts/
  run_cade_all.sh    CADE for every SDGBiasBench context test of a list of models
```



## Installation

```bash
git clone https://github.com/jeffreylin122/SDGBiasBench.git
cd SDGBiasBench
pip install -r requirements.txt
```



## Preparing SDGBiasBench

```bash
huggingface-cli download jeffreylin1222/SDGBiasBench --repo-type dataset --local-dir data/SDGBiasBench
python -m sdgbiasbench.make_views --data-root data/SDGBiasBench --out views
```

This writes 14 files to `views/`: `mcq_T1 … mcq_T5`, `regression_T2`, `regression_T5`, and a
`_text` version of each with a blank image. Every question has an `index` that is the same in
all of its views, plus `split`, `answer` and, for MCQ, the options `A`–`C`. Together the files
provide the four evidence views of every context test (see the view table above).

## CADE

CADE works on any task where the model can be queried under four views of each question:


| view          | model input                |
| ------------- | -------------------------- |
| Full          | image + context + question |
| Image-only    | image + question           |
| Context-only  | context + question         |
| Question-only | question                   |


**1. Collect the model outputs.** For every question and view, record the model's generated
answer and the logit of each answer candidate at the **first generated token**. MCQ candidates
are the option letters; for numeric answers they are the digits 0–9.

**2. Arrange one table per view** (`.csv`, `.parquet` or `.xlsx`, one row per question):


| column       | tables               | content                                                                                                        |
| ------------ | -------------------- | -------------------------------------------------------------------------------------------------------------- |
| `index`      | all four             | question id, the same in every view                                                                            |
| `logit_<c>`  | all four             | first-token logit of candidate `c`: `logit_A`, `logit_B`, … (MCQ) or `logit_0` … `logit_9` (numeric)           |
| `prediction` | Full                 | the model's generated answer                                                                                   |
| `answer`     | Full                 | gold answer: an option letter, or a number                                                                     |
| `split`      | Full, optional       | `val` rows tune CADE, `test` rows are reported. Without it, 20% of the rows (by a hash of `index`) form `val`. |
| `A`, `B`, …  | Full, optional (MCQ) | option texts; an empty cell marks an option the question does not have                                         |


Save the four tables in one folder, named after the task (`mcq` or `regression`):

| view | table name |
|---|---|
| Full | `<task>_T1` |
| Context-only | `<task>_T1_text` |
| Image-only | `<task>_T5` |
| Question-only | `<task>_T5_text` |

SDGBiasBench is prepared the same way. Run the model on each view file and save the table under
the view file's name, e.g. `preds/<model>/mcq_T1.csv`, keeping the question's `index`, `split`,
`answer` and options. Its other context tests follow the same pattern with `T<K>` in place of
`T1` (`--level K`).

**3. Run CADE.**

```bash
# MCQ (any number of option letters): reads mcq_T1, mcq_T1_text, mcq_T5, mcq_T5_text
python -m cade.run_mcq search --pred-dir preds/<model> --out-dir runs/<model>/mcq --save-preds

# numeric answers; --tolerance sets the interval-accuracy margin
python -m cade.run_regression search --pred-dir preds/<model> --tolerance 1.0 --out-dir runs/<model>/regression

# SDGBiasBench: other context tests with --level (regression has T2 and T5), or all at once
python -m cade.run_mcq search --pred-dir preds/<model> --level 3 --out-dir runs/<model>/mcq_T3
python -m cade.run_regression search --pred-dir preds/<model> --level 2 --out-dir runs/<model>/regression_T2
bash scripts/run_cade_all.sh preds runs <model>

# re-apply tuned hyperparameters to new predictions
python -m cade.run_mcq apply --pred-dir preds/<new_model> --params runs/<model>/mcq/cade_config.json
```

`search` tunes CADE's hyperparameters on the val split and reports on the test split:

- **MCQ:** accuracy of the baseline and of CADE. 
- **Numeric answers:** MAE and interval accuracy.

`--out-dir` saves `cade_config.json` (hyperparameters and metrics), and `--save-preds` also
writes `cade_predictions.csv` with the baseline and CADE answer of every question. Run
`python -m cade.run_mcq search --help` for all options.

**In Python**, given `(N, O)` first-token logit tensors of the four views:

```python
from cade import CADEParams, prepare_views, predict, random_search

pv = prepare_views({"full": z_full, "img": z_img, "ctx": z_ctx, "q": z_q},
                   mask=valid,        # (N, O) bool: options each question has (optional)
                   labels=y)          # (N,) gold candidate index (needed only for tuning)
best = random_search(pv, val_rows).best     # val_rows: LongTensor of tuning-question indices
pred = predict(pv, best)[0]                 # (N,) debiased candidate index

pred = predict(pv, CADEParams(alpha=1.0, lambda_kl=2.0, beta=1.0, tau=0.9))[0]  # fixed hyperparameters
```



## Citation

```bibtex
@article{lin2026sdgbiasbench,
  title={SDGBiasBench: Benchmarking and Mitigating Vision--Language Models' Biases in Sustainable Development Goals},
  author={Lin, Zihang and Qin, Huaiyuan and Yang, Muli and Zhu, Hongyuan},
  journal={arXiv preprint arXiv:2605.21919},
  year={2026}
}
```



## License

The code is released under the [MIT License](LICENSE). The dataset is released under CC BY-SA 4.0 (see the [dataset card](https://huggingface.co/datasets/jeffreylin1222/SDGBiasBench)).
