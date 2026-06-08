# rf_combo_nofoldseek

**Pooled OOF CLS:** `0.0000` (PR-AUC `0.2862`, WSpearman `0.0000`)

vs Mandrake baseline `0.318`: **-0.3180**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.5833 |
| LTR_Retrotransposon | 11 | 2 | 0.1964 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.3371 |
| Retroviral | 18 | 12 | 0.5533 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 52 unique values; std `0.1949`; degenerate: `False`
- **Class vs Rank gap:** 0.2862 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `1.0000` (NOT significant vs label-shuffle null; null mean `0.1152`)