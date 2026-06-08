# A_oracle_binary

**Pooled OOF CLS:** `0.2385` (PR-AUC `1.0000`, WSpearman `0.1354`)

vs Mandrake baseline `0.318`: **-0.0795**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 1.0000 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 1.0000 |
| Retroviral | 18 | 12 | 1.0000 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 2 unique values; std `0.4824`; degenerate: `True`
- **Class vs Rank gap:** 0.8646 (classifier-only: `True`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.2540` (NOT significant vs label-shuffle null; null mean `0.1206`)