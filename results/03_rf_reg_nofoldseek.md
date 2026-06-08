# rf_reg_nofoldseek

**Pooled OOF CLS:** `0.1275` (PR-AUC `0.3504`, WSpearman `0.0779`)

vs Mandrake baseline `0.318`: **-0.1905**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.5833 |
| LTR_Retrotransposon | 11 | 2 | 0.2667 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.4800 |
| Retroviral | 18 | 12 | 0.5602 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 57 unique values; std `2.2503`; degenerate: `False`
- **Class vs Rank gap:** 0.2724 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.4020` (NOT significant vs label-shuffle null; null mean `0.1241`)