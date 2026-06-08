# lgbm_clf_nofoldseek

**Pooled OOF CLS:** `0.2725` (PR-AUC `0.2657`, WSpearman `0.2797`)

vs Mandrake baseline `0.318`: **-0.0455**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.3667 |
| LTR_Retrotransposon | 11 | 2 | 0.1833 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.3813 |
| Retroviral | 18 | 12 | 0.6476 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 54 unique values; std `0.2762`; degenerate: `False`
- **Class vs Rank gap:** 0.0140 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.2180` (NOT significant vs label-shuffle null; null mean `0.1228`)