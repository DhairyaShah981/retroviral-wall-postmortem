# lgbm_reg_allfeat

**Pooled OOF CLS:** `0.2401` (PR-AUC `0.4635`, WSpearman `0.1620`)

vs Mandrake baseline `0.318`: **-0.0779**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.3929 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.3717 |
| Retroviral | 18 | 12 | 0.6038 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 57 unique values; std `4.3752`; degenerate: `False`
- **Class vs Rank gap:** 0.3015 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.2580` (NOT significant vs label-shuffle null; null mean `0.1181`)