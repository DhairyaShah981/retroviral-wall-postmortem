# B_family_constant

**Pooled OOF CLS:** `0.1046` (PR-AUC `0.6054`, WSpearman `0.0572`)

vs Mandrake baseline `0.318`: **-0.2134**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.4000 |
| LTR_Retrotransposon | 11 | 2 | 0.1818 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.4167 |
| Retroviral | 18 | 12 | 0.6667 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 4 unique values; std `0.3973`; degenerate: `False`
- **Class vs Rank gap:** 0.5482 (classifier-only: `True`)
- **Family-constant predictions:** `True` (min within-family std `0.0000`)
- **Permutation p-value:** `0.4080` (NOT significant vs label-shuffle null; null mean `0.1149`)