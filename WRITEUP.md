# Reading the Retroviral Wall

*A 1-week post-mortem on Mandrake Bio's Open Problem #1, and the eval harness I'd want for Open Problems #2-N.*

— Dhairya Shah · June 2026 · [github.com/DhairyaShah981/retroviral-wall-postmortem](https://github.com/DhairyaShah981/retroviral-wall-postmortem)

---

## TL;DR

I spent a week on Mandrake's Retroviral Wall Challenge as a post-mortem (Stage 1 closed April 30; you're sorting Stage 2 results now). The headline result is **a methodology-audit harness** that flags the kinds of submissions tied at 1.0 on your Kaggle public leaderboard — and along the way, **CLS 0.544 on LOFO** (vs your published reference of 0.318) using a 60-line isotonic-calibrated RF stack on the provided handcrafted features.

Four findings worth your attention:

1. **The wall isn't Retroviral. It's LTR_Retrotransposon.** Held-out Retroviral PR-AUC is 0.83 across reasonable models. LTR's is 0.27, consistently, across classification, regression, and cross-family triage. The challenge name foregrounds Retroviral because of MMLV, but the real held-out failure is on the minority-active sparse family.

2. **The Kaggle leaderboard is methodologically unsortable.** Top 6 entries tied at 1.00000 on a 57-sample LOFO problem with the page-spec CLS is a tell — the public metric is almost certainly simpler than CLS. Stage 2 wet-lab exists to sort them. My audit harness sorts them dry-side, before wet-lab cost.

3. **Your page and your scorer disagree on ε.** Page text says `weight_i = pe_efficiency_i + ε (ε = 0.1)`. `evaluation/evaluate.py` uses `EPSILON = 0.01`. 10× difference on inactive-RT weight; the audit harness reports both.

4. **Sequence-identity proximity probe confirms the model isn't memorizing.** Spearman r(nearest-cross-family-identity, prediction-error) = **−0.06** — i.e., no correlation. The model is not doing nearest-neighbour lookup. Separately, LTR has the **lowest mean cross-family identity (34.7%)** of any active-containing family — that's the *structural* reason it's the wall: training on Retroviral RTs gives the model very little to lean on when predicting LTR. A real signal would have to come from a feature space orthogonal to evolutionary distance.

The package — `mandrake-bench` — is open source (MIT, `pip install -e .`) with a LOFO loop, the official CLS, four methodology-audit detectors, a sequence-identity proximity probe, and a cross-family transfer-triage selector for the wet-lab loop.

---

## What I built

`mandrake-bench` is small on purpose. Five modules:

- **`metrics`** — CLS = harmonic mean of PR-AUC + Weighted Spearman, matching `evaluation/evaluate.py` byte-for-byte (verified to 0.006 due to sklearn argsort tie-breaking). Exposes both `EPSILON_CODE = 0.01` and `EPSILON_PAGE = 0.1` so submissions can be scored against either spec.
- **`cv.lofo_predict`** — Leave-One-Family-Out cross-validation loop. Pluggable `fit_predict_fn` signature; supports passing PE-efficiency to regression-target models.
- **`baselines`** — Reference baselines: predict-all-inactive (CLS 0.000), random (0.31), family-prior (constant prior), ESM-only LogReg (the family-leak canary at CLS 0.18), RF clf, RF reg.
- **`audit`** — Four methodology detectors:
  - `degeneracy` — flags submissions with ≤2 unique values (PR-AUC=1.0 binary games)
  - `family_leakage` — flags submissions with within-family std = 0 (memorising family ID)
  - `class_rank_consistency` — flags PR-AUC strong + WSpearman weak (classifier-only, won't rank within actives for wet-lab triage)
  - `shuffle_null_cls` — permutes labels 5000× and reports p-value vs observed CLS (catches noise predictors riding class imbalance; add-one-smoothed so p never reads 0)
- **`active.next_batch`** — RF-tree-std uncertainty × predicted efficiency selector for wet-lab triage. Exposes a `regret_simulation` that scores capture rates on held-out folds.

There's also a `structural` module that extracts 12 family-agnostic geometry features from ESMFold PDBs (YXDD-motif radius-of-gyration, catalytic-pocket volume proxy, aromatic/charged contact counts, palm-thumb domain distance). I tried it. **It doesn't help** — the existing 66 handcrafted features already capture this signal. Finding worth reporting.

---

## Results

All LOFO pooled OOF, scored with the official CLS, ε=0.01.

| Model | CLS | PR-AUC | WSpearman | Retroviral PR-AUC | LTR PR-AUC |
|---|---|---|---|---|---|
| **isotonic_stack_handonly** | **0.544** | 0.600 | 0.498 | 0.827 | 0.268 |
| rf_simple_handonly | 0.520 | 0.516 | 0.526 | 0.836 | 0.208 |
| rf_simple_combined (+ struct) | 0.430 | 0.367 | 0.519 | 0.787 | 0.163 |
| Mandrake reference (Handcrafted + RF) | 0.318 | — | — | — | — |
| LightGBM classifier (overfits N=57) | 0.303 | 0.343 | 0.272 | 0.743 | 0.225 |
| Random | 0.313 | 0.571 | 0.215 | — | — |
| ESM-only LogReg (family-leak canary) | 0.184 | 0.381 | 0.121 | — | — |
| Predict-all-inactive | 0.000 | 0.368 | 0.000 | — | — |

The winning model is **embarrassingly simple**: RF classifier (proba) + RF regressor (PE efficiency) on the handcrafted features, isotonic-calibrated, geometric-meaned. 60 lines of Python. The reason it beats your published reference is the calibration + the matching of the two heads to PR-AUC and WSpearman respectively. Most participants seem to train one model end-to-end and miss that CLS is harmonic-meaned across two separate objectives.

---

## Five findings worth your team's attention

### 1. The wall is LTR_Retrotransposon, not Retroviral

Across every reasonable model, the held-out fold breakdown looks like this:

```
Retroviral       PR-AUC 0.54 – 0.84
Retron           PR-AUC 0.49 – 0.93
Group_II_Intron  PR-AUC 0.37 – 1.00
LTR              PR-AUC 0.16 – 0.39  ← consistently weakest
```

The challenge framing foregrounds Retroviral because (a) it contains 12 of 21 active RTs, (b) it contains MMLV, the gold-standard at 41% efficiency. But LOFO doesn't care about absolute family size — it cares about whether you can predict held-out activity from the *other* families. Retroviral has 12 actives to learn from in adjacent folds. LTR_Retrotransposon has 2 actives across 11 candidates, of which the held-out is held out — so models effectively train on 0-2 actives in the LTR family and try to recall them. The wall is the *minority-active sparse family*, not the *gold-standard family*.

This matters operationally because your wet-lab triage is going to need different strategies for the two cases — for Retroviral you have abundant training signal and need precision; for LTR you have nothing and need exploration.

### 2. RF beats LightGBM at N=57

Vanilla `RandomForestClassifier(n_estimators=300)` gets CLS 0.52. LightGBM with comparable depth/leaves gets CLS 0.30. With N=57, the gradient-boosting iteration count is bigger than the dataset; it overfits to whichever tree happens to fit the held-out fold worst. RF's bagging averages this out. The Mandrake reference baseline of 0.318 is suspiciously close to my LightGBM number — it might be that the published reference was over-tuned on a single fold, or trained with too many estimators relative to the data.

Practical implication: don't believe any submission whose CLS gain came from "trying a fancier gradient booster." It's a noise generator at this N.

### 3. The page vs code ε discrepancy

```python
# Page text (mandrake.bio/retroviral-wall-challenge/):
#   weight_i = pe_efficiency_i + ε  (ε = 0.1)
# evaluation/evaluate.py line 14:
#   EPSILON = 0.01
```

With ε=0.01, inactive RTs get weight 0.01, MMLV (41%) gets weight 41.01 — a 4101× ratio. With ε=0.1, the ratio is 411×. This changes how harshly the model is penalized for misranking an inactive. Pick one and document it on the page. The audit harness scores against both so participants don't get bitten.

### 4. foldseek_TM_* features carry signal + leak

Of the 66 handcrafted features, 10 are `foldseek_TM_*` (TM-scores to reference RT crystal structures across families). These are *biophysically meaningful* (they encode real structural similarity) but also *family-leaky* (they encode family identity by construction).

I tried dropping them. **CLS drops by ~0.07.** They carry real signal mixed with leakage. The right move isn't removal — it's a leakage-aware regularizer that uses them only insofar as they predict activity orthogonal to family.

For Open Problem #2: ship the dataset with a "leaky features" column in the feature dictionary, plus a `mandrake_bench.audit.leakage_aware_importance()` check that flags submissions whose top-importance features overlap the leaky-features list.

### 5. ESM-2 PCA-orthogonal projection kills WSpearman

I tried debiasing ESM-2 embeddings by projecting onto the orthogonal complement of the top-K principal components (which empirically span the family-mean subspace). Result: per-family PR-AUC stays decent (0.6-0.8), but pooled Weighted Spearman collapses to 0. The model classifies but doesn't rank within the active set.

This is exactly the failure mode `audit.class_rank_consistency` flags. It's an excellent natural example of why CLS is harmonic-meaned across two objectives — a model that aces one and tanks the other gets caught.

---

## Why simple beats deep at N=57 — and what to read on it

The empirical result (RF 0.52 vs LightGBM 0.30 vs ESM-only LogReg 0.18) is the standard small-N protein-engineering regime predicted in [Hsu, Verkuil, Liu et al. 2022](https://pubmed.ncbi.nlm.nih.gov/35039677/) — *"Learning protein fitness models from evolutionary and assay-labeled data."* Their result: on small assay-labelled datasets, ridge regression on one-hot site features + one evolutionary-density feature (EVmutation or Potts) beats deep models. The intuition: 57 samples is below the inflection point where self-supervised representations help; supervised, regularized models with handcrafted biophysics dominate.

What this means for *Open Problem #2*: if the next dataset is going to be a similar size, ship a "boring baselines" section in the challenge README so participants don't waste cycles fine-tuning ESM-3B. And ship `mandrake-bench` as the participant-facing benchmark so they're optimising against the same eval surface you score against internally.

A few approaches I'd want to try with more time, with citations so they're not vibes:

- **Supervised contrastive ESM fine-tune (CLEAN-style; Yu et al. 2023, *Science*, [doi:10.1126/science.adf2465](https://www.science.org/doi/10.1126/science.adf2465)).** Same-activity RTs as positives, different-activity-or-family as negatives, family-balanced batches. The principled version of the ESM-PCA-orthogonal hack I tried and abandoned.
- **ProteinMPNN-derived per-residue log-likelihoods over the YXDD catalytic motif** as features. [Liu et al. 2026, *Nature Biotechnology* s41587-026-03149-6](https://www.nature.com/articles/s41587-026-03149-6) used inverse-folding redesign to engineer PE8/PEmax to 2.9× efficiency in mice — same group, same RT domain, different question. Inverse folding is an engineering tool; its log-likelihoods can be a classification feature.
- **The Doman et al. 2023 paper that built this dataset** ([Cell, S0092-8674(23)00854-1](https://www.cell.com/cell/fulltext/S0092-8674(23)00854-1)) reports per-family efficiencies — useful prior for active-class weighting that I haven't fully exploited.

---

## Cross-family transfer triage: works for 3 of 4 active-containing families

> Naming note: this is **not** a full acquire-fit-acquire active-learning loop — that requires a Stage 2 wet-lab feedback channel. It's a one-shot triage selector: model fitted on N-1 families, top-K candidates from the held-out family go to the wet lab first.

Given a fitted RF regressor on PE efficiency, I extract RF-tree-std as a predictive uncertainty estimate, then select top-K candidates by `(predicted_efficiency**α * uncertainty**β)` with α=2, β=1. Simulating wet-lab triage on each held-out family:

| Held-out family | n | active | k → lab | AL capture | Random | Lift |
|---|---|---|---|---|---|---|
| Group_II_Intron | 5 | 2 | 1 | **93%** | 19% | **+73 pp** |
| Retron | 12 | 5 | 4 | 61% | 35% | +26 pp |
| Retroviral | 18 | 12 | 5 | 35% | 28% | +7 pp |
| LTR_Retrotransposon | 11 | 2 | 3 | 0% | 28% | **-28 pp** |

Two real signals:
- The selector works on families with diffuse active-rate (Retron, Group_II_Intron).
- It fails on LTR — the same family that's the held-out wall. This isn't a selector bug; it's a *feature-space* failure. The biophysics features in `handcrafted_features.csv` don't carry LTR-activity signal in any regime — classification, ranking, or active selection. Real wet-lab signal must come from somewhere outside this 66-D feature space (maybe sequence-context-dependent, maybe RNA-template-binding, maybe Cas9-fusion compatibility).

That's a useful negative result for Open Problem #2 scoping: don't expect handcrafted-feature ML to crack LTR. Either change the feature space or change the family weighting.

---

## Sequence-identity proximity — the model isn't memorizing

A standard CASP-style rigour check: for every RT, compute max %identity to the nearest RT in *any other family* (simulating what training would see during LOFO). Then plot prediction error vs that identity. If models learned nearest-neighbour lookup, errors would collapse for high-identity points.

The result (`notebooks/06_identity_proximity.py`):

```
Spearman r(nearest_cross_family_identity, prediction_error) = -0.06
```

Effectively zero. The isotonic-stack model is **not** doing nearest-neighbour lookup — it's actually learning from the handcrafted biophysics across families.

But the per-family identity stats reveal the deeper *structural* reason the LTR wall exists:

| Family | n | Mean max-cross-family-identity |
|---|---|---|
| CRISPR-associated | 5 | 38.4% |
| Group_II_Intron | 5 | 37.5% |
| Other | 5 | 36.8% |
| Retron | 12 | 35.9% |
| Retroviral | 18 | 35.6% |
| **LTR_Retrotransposon** | **11** | **34.7%** ← lowest |

LTR_Retrotransposon RTs are the **most evolutionarily distant from active RTs in other families**. When LTR is held out, the model is asked to predict activity from training examples that share, on average, only ~35% sequence identity — twilight-zone homology. There simply isn't enough sequence proximity for the biophysical features to interpolate from. **Real LTR-activity signal must live in a feature space orthogonal to evolutionary distance** — sequence-context-dependent, or in the Cas9-fusion interface, or in RNA-template-binding kinetics.

That's a dataset-level finding, not a model-level one. For Open Problem #2: if a target family has mean cross-family identity below ~35%, no biophysical-features ML is going to crack it without new features.

---

## The audit harness on synthetic submissions

I built three synthetic submissions designed to game the metric, to demonstrate the audit catches them:

| Submission | CLS | PR-AUC | WSpearman | Audit verdict |
|---|---|---|---|---|
| **A: Oracle binary** (predict y_true) | 0.239 | **1.000** | 0.135 | flagged `classifier-only` (gap 0.86) |
| **B: Family-constant** (1.0 Retroviral, 0.5 Retron, 0.1 LTR, 0.0 rest) | 0.105 | 0.605 | 0.057 | flagged `family-constant` (within-fam std 0.0) + `classifier-only` |
| **C: Oracle ranker** (predict pe_efficiency_pct) | **1.000** | **1.000** | **1.000** | clean — shuffle p < 2e-4 SIGNIFICANT |

A binary oracle that matches PR-AUC 1.0 collapses to CLS 0.24. The audit flags it before you'd waste wet-lab capacity on it. The family-constant attacker is caught by `family_leakage`. The true oracle ranker passes all checks. That's the harness doing its job.

If you ran this audit against your current 6 Kaggle leaderboard entries, it would tell you which of them is a binary classifier riding the imbalance, which is family-memorising, and which (if any) actually ranks within the active set. That's the dry-side answer to a problem you bolted Stage 2 onto solve.

---

## What I'd build at Mandrake

The package I shipped is a participant-facing harness. The natural follow-on is the Mandrake-internal version:

1. **`mandrake_bench.kaggle.sync()`** — pull leaderboard entries + their predictions, auto-audit them all, surface ranked findings.
2. **Per-Open-Problem dataset card auto-generator** — given a new dataset with family annotations, produce a participant-facing README + a leaky-features list + the suggested LOFO splits.
3. **Wet-lab Stage 2 ranking module** — given participant predictions on Stage 1 + Stage 2 candidates, produce a regret curve showing capture rate vs k for each participant. The team you hire is the one with the steepest curve, not the highest pooled CLS.
4. **A "future problems" template** — copy this repo's structure, swap the dataset, change the family-grouping function, you've got Open Problem #2 ready to launch.

That's three weekends of work and it turns each future Open Problem into a one-week setup instead of a multi-week one.

---

## What I won't claim

- **I'm not a senior protein-ML researcher.** I joined ML LLM-first. I don't have the pre-LLM intuition about loss landscapes the Sr. AI Engineer JD asks for.
- **My structural features added noise**, not signal. Catalytic-site geometry is already in your handcrafted set.
- **My biggest single CLS gain came from a 60-line isotonic stack**, not from a novel architecture. The right move at N=57 is simple+rigorous, not deep.
- **My LTR results are negative.** I couldn't crack it. I don't know if it's crackable from the features Mandrake provided. That's a real open question.

What I bring is the engineer who builds the surface around the modeller's models. The eval harness, the audit, the active-learning loop, the participant-facing tooling that makes each future Open Problem run cleanly. Voice-eval-harness for LLMs at Trifetch is the same shape. healthcare-skills (MIT, npm) and bahmni-ai (FHIR/RAG) are the same shape. This post-mortem is the same shape applied to your domain.

---

## How to reproduce

```bash
git clone https://github.com/DhairyaShah981/retroviral-wall-postmortem
cd retroviral-wall-postmortem
git clone https://github.com/Mandrake-Bioworks/Retroviral-Wall-Challenge.git mandrake-repo
ln -s "$(pwd)/mandrake-repo/data" data
python3 -m venv .venv && source .venv/bin/activate
pip install -e . lightgbm
python notebooks/01_baselines.py     # baselines + family-leak canary
python notebooks/04_day2_stack.py    # isotonic stack + transfer triage
python notebooks/05_audit_synthetic_perfect.py  # audit on synthetic gamed submissions
python notebooks/06_identity_proximity.py       # sequence-identity proximity probe
python mandrake-repo/evaluation/evaluate.py \
    --predictions results/04_isotonic_stack_handonly_predictions.csv
```

Everything is CPU. No GPU. Full pipeline runs in under 3 minutes on a MacBook.

---

## References

- Doman, J. L. et al. (2023). Phage-assisted evolution and protein engineering yield compact, efficient prime editors. *Cell.* [S0092-8674(23)00854-1](https://www.cell.com/cell/fulltext/S0092-8674(23)00854-1). *(The dataset source.)*
- Liu, J. et al. (2026). AI-guided redesign of laboratory-evolved reverse transcriptases enhances prime editing. *Nature Biotechnology.* [s41587-026-03149-6](https://www.nature.com/articles/s41587-026-03149-6). *(David Liu group's RT redesign with ProteinMPNN.)*
- Hsu, C., Verkuil, R., Liu, J. et al. (2022). Learning protein fitness models from evolutionary and assay-labeled data. *Nature Biotechnology.* [PMID 35039677](https://pubmed.ncbi.nlm.nih.gov/35039677/). *(Why simple supervised models beat deep at small N.)*
- Yu, T., Cui, H., Li, J. C. et al. (2023). Enzyme function prediction using contrastive learning. *Science.* [doi:10.1126/science.adf2465](https://www.science.org/doi/10.1126/science.adf2465). *(CLEAN — supervised contrastive for enzyme function.)*
