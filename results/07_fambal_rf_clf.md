# fambal_rf_clf

**Pooled OOF CLS:** `0.4667` (PR-AUC `0.5002`, WSpearman `0.4374`)

vs Mandrake baseline `0.318`: **+0.1487**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.1964 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.6587 |
| Retroviral | 18 | 12 | 0.7963 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 57 unique values; std `0.1217`; degenerate: `False`
- **Class vs Rank gap:** 0.0629 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0140` (SIGNIFICANT vs label-shuffle null; null mean `0.1210`)