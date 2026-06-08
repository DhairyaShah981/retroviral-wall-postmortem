# rf_clf_allfeat

**Pooled OOF CLS:** `0.4651` (PR-AUC `0.3995`, WSpearman `0.5566`)

vs Mandrake baseline `0.318`: **+0.1471**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.2083 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.5852 |
| Retroviral | 18 | 12 | 0.8334 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 57 unique values; std `0.1338`; degenerate: `False`
- **Class vs Rank gap:** 0.1571 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0240` (SIGNIFICANT vs label-shuffle null; null mean `0.1251`)