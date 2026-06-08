# Retroviral Wall — Post-Mortem + `mandrake-bench`

A 1-week post-mortem of [Mandrake Bio's Open Problem #1: The Retroviral Wall](https://www.mandrake.bio/retroviral-wall-challenge/) +
the eval/audit harness Mandrake (or any future participant) can reuse for Open Problem #2.

> Stage 1 of the Kaggle challenge closed April 30, 2026. Stage 2 wet-lab validation is happening Q2 2026.
> This work is a **post-mortem, not a competition entry**.

---

## Status snapshot (Day 0/1 — 2026-06-07)

### What's running end-to-end already
- Repo cloned: `mandrake-repo/` (full dataset + official evaluator)
- Python 3.13 venv with numpy / pandas / sklearn / lightgbm / xgboost / biopython / kaggle
- `mandrake_bench` package skeleton — `metrics`, `cv`, `baselines`, `audit`, `report`
- 3 notebooks reproducing + extending baselines (`notebooks/01_*.py`, `02_*.py`, `03_*.py`)
- Per-model markdown + CSV outputs in `results/`
- CLS impl verified against official `mandrake-repo/evaluation/evaluate.py` (0.006 sklearn tie-break gap)

### Headline numbers (pooled OOF, LOFO)
| Model | CLS | PR-AUC | WSpearman | Retroviral PR-AUC | LTR PR-AUC |
|---|---|---|---|---|---|
| **RF classifier on handcrafted (simple)** | **0.534** | 0.531 | 0.526 | **0.836** | 0.208 |
| Mandrake reference (Handcrafted + RF) | 0.318 | — | — | — | — |
| RF clf, 600 trees + tuned | 0.465 | 0.399 | 0.557 | 0.833 | 0.208 |
| RF reg on PE efficiency | 0.465 | 0.500 | 0.435 | 0.604 | 0.393 |
| LGBM clf (overfits at N=57) | 0.303 | 0.343 | 0.272 | 0.743 | 0.225 |
| ESM-only LogReg (family-leak canary) | 0.184 | 0.381 | 0.121 | — | — |
| Predict-all-inactive | 0.000 | 0.368 | 0.000 | — | — |

### Novel findings (already shippable as talking points to Tanay)
1. **The wall isn't Retroviral — it's LTR_Retrotransposon.** Every model's Retroviral held-out PR-AUC is 0.54-0.84. LTR's stays 0.17-0.27. The challenge page foregrounds Retroviral because it has the most active RTs and the gold-standard MMLV — but the *generalization collapse* is on LTR (2/11 active, sparse-positive minority class).
2. **RF beats LightGBM decisively (0.53 vs 0.30) at N=57.** Standard gradient-boosting overfits this regime. The Mandrake reference 0.318 may be from over-tuning; simpler models win.
3. **Methodology bug found:** challenge page says `ε = 0.1` for Weighted Spearman weights; official `evaluation/evaluate.py` uses `ε = 0.01`. 10x difference. We follow the code (it's what gets scored) and expose both via kwarg.
4. **Dropping `foldseek_TM_*` columns** (10 explicit family-similarity features) actually *hurts* pooled CLS — they carry real signal mixed with leak. Pure-biophysics models (no FoldSeek) drop ~0.07 CLS. So the "ideal" model needs FoldSeek + a leakage-aware regularizer, not just removal.
5. **ESM-2 LogReg with PCA-orthogonal-projection** kills Weighted Spearman to 0 even with high per-family PR-AUC — i.e., it classifies but doesn't rank. Caught by the `audit.class_rank_consistency()` check, which is exactly the kind of submission Mandrake needs to flag.

### What the audit harness already detects
- **Score degeneracy** — flags submissions with ≤2 unique values (classifies but won't rank)
- **Class vs Rank gap** — flags submissions that nail PR-AUC but tank WSpearman (binary classifiers gaming imbalanced data)
- **Family-constant predictions** — flags submissions where within-family std is ~0 (memorising family ID)
- **Shuffle null** — permutes labels 500x and reports p-value vs the observed CLS (catches noise predictors that ride class imbalance)

These four checks would together separate the 6 Kaggle entries currently tied at 1.00000.

---

## Repo layout

```
retroviral-wall-postmortem/
├── README.md                       # this file
├── .venv/                          # python 3.13 venv
├── data → mandrake-repo/data       # symlink
├── mandrake-repo/                  # upstream challenge repo
│   ├── data/
│   │   ├── rt_sequences.csv        # 57 RTs: sequence, active, pe_efficiency_pct, rt_family
│   │   ├── handcrafted_features.csv# 66 biophysical features
│   │   ├── esm2_embeddings.npz     # (57, 1280) ESM-2 mean-pooled
│   │   ├── feature_dictionary.csv  # per-feature descriptions
│   │   └── structures/             # 57 ESMFold PDB files
│   └── evaluation/evaluate.py      # official CLS scorer
│
├── mandrake_bench/                 # MY reusable package
│   ├── metrics.py                  # CLS (matches official, both ε values exposed)
│   ├── cv.py                       # LOFO cross-validation loop
│   ├── baselines.py                # all-inactive, random, family-prior, RF, LR
│   ├── audit.py                    # 4 leakage / methodology detectors
│   └── report.py                   # markdown rendering matching Mandrake's page format
│
├── notebooks/                      # executable .py scripts (not .ipynb to keep diff-friendly)
│   ├── 01_baselines.py             # reproduce CLS 0.318 reference + family-leak canary
│   ├── 02_novel_model.py           # LightGBM ablations + ESM PCA-orthogonal experiments
│   └── 03_smart_rf.py              # RF hyperparam sweep + combo + ESM-PCA-clean ensemble
│
└── results/                        # auto-generated per run
    ├── 0X_<model>.md               # per-model markdown report + audit
    ├── 0X_<model>_predictions.csv  # submission-format predictions
    └── 0X_summary.csv              # comparison table
```

---

## Quick start

```bash
cd ~/Documents/retroviral-wall-postmortem
source .venv/bin/activate
python notebooks/01_baselines.py    # baselines
python notebooks/02_novel_model.py  # LightGBM ablations
python notebooks/03_smart_rf.py     # RF + ESM-PCA experiments

# Validate any submission against the official evaluator
python mandrake-repo/evaluation/evaluate.py \
    --predictions results/01_rf_handcrafted_clf_predictions.csv
```

---

## What's next (Day 2-7 of the sprint plan)

See [vault: `15 - Mandrake Tanay Call 1-Week Sprint`](../../Library/Mobile%20Documents/iCloud~md~obsidian/Documents/dhiru-brain/dhiruxd/career/15%20-%20Mandrake%20Tanay%20Call%201-Week%20Sprint.md) for the full plan.

**Immediate Day 2 priorities:**
- [ ] **Push CLS past 0.55** — try stacking RF clf + RF reg with proper isotonic calibration, not naive geometric mean
- [ ] **Add hand-crafted structural features** from `data/structures/*.pdb` using Biopython — DSSP secondary structure on YXDD motif neighborhood, contact map within catalytic site
- [ ] **Active-learning module** (`mandrake_bench.active.next_batch`) — given a fitted model + candidate pool, return top-N candidates by efficiency × uncertainty
- [ ] **`audit.shuffle_null_cls` n_shuffles → 5000** + cache results so the audit reports include hard p-values

**Day 3-4:**
- [ ] Package `mandrake_bench` with `pyproject.toml`, type hints, tests, README
- [ ] Build the writeup notebook: "Reading the Retroviral Wall — what I learned trying to break it in a week"
- [ ] Run the audit harness against the 4 currently-best models and a synthetic "tied at 1.00000" submission to show the audit catches what the leaderboard can't

**Day 5-6:**
- [ ] 8-min Loom walkthrough (intro → results → harness API → pitch for Open Problem #2)
- [ ] Publish repo on GitHub (`dhairyashah/retroviral-wall-postmortem`)
- [ ] DM Tanay before Wednesday call

---

## What we know about the leaderboard

Top 6 Kaggle entries all scored **exactly 1.00000**. On a 57-sample LOFO dataset, that's a methodology tell, not a winning signal — likely from a simpler Kaggle public metric (probably classification PR-AUC or accuracy, not the page-spec CLS) plus optimal classifier behavior on the imbalanced class structure.

Mandrake's Stage 2 wet-lab validation on ~40 fresh RT candidates exists *precisely* to sort these tied teams. The audit harness here is the dry-side tool that would have flagged the same models before they hit Stage 2.

---

## License

MIT (TBD on GitHub publish). Dataset © Mandrake Bio per their challenge terms.
