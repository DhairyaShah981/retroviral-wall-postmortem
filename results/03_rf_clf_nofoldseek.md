# rf_clf_nofoldseek

**Pooled OOF CLS:** `0.2971` (PR-AUC `0.2541`, WSpearman `0.3575`)

vs Mandrake baseline `0.318`: **-0.0209**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.5000 |
| LTR_Retrotransposon | 11 | 2 | 0.1714 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.3646 |
| Retroviral | 18 | 12 | 0.5417 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 56 unique values; std `0.1399`; degenerate: `False`
- **Class vs Rank gap:** 0.1034 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.1860` (NOT significant vs label-shuffle null; null mean `0.1240`)