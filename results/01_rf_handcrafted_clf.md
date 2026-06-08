# rf_handcrafted_clf

**Pooled OOF CLS:** `0.5285` (PR-AUC `0.5314`, WSpearman `0.5255`)

vs Mandrake baseline `0.318`: **+0.2105**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.2083 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.7052 |
| Retroviral | 18 | 12 | 0.8365 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 47 unique values; std `0.1566`; degenerate: `False`
- **Class vs Rank gap:** 0.0059 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0020` (SIGNIFICANT vs label-shuffle null; null mean `0.1235`)