# C_oracle_ranker

**Pooled OOF CLS:** `1.0000` (PR-AUC `1.0000`, WSpearman `1.0000`)

vs Mandrake baseline `0.318`: **+0.6820**


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

- **Score degeneracy:** 20 unique values; std `9.2798`; degenerate: `False`
- **Class vs Rank gap:** 0.0000 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0000` (SIGNIFICANT vs label-shuffle null; null mean `0.1240`)