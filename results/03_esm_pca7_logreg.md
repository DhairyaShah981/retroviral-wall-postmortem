# esm_pca7_logreg

**Pooled OOF CLS:** `0.0000` (PR-AUC `0.3807`, WSpearman `0.0000`)

vs Mandrake baseline `0.318`: **-0.3180**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.8333 |
| LTR_Retrotransposon | 11 | 2 | 0.3667 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.4998 |
| Retroviral | 18 | 12 | 0.6411 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 50 unique values; std `0.4160`; degenerate: `False`
- **Class vs Rank gap:** 0.3807 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `1.0000` (NOT significant vs label-shuffle null; null mean `0.1041`)