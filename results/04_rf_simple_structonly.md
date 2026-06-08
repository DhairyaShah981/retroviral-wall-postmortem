# rf_simple_structonly

**Pooled OOF CLS:** `0.2589` (PR-AUC `0.2579`, WSpearman `0.2599`)

vs Mandrake baseline `0.318`: **-0.0591**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.5000 |
| LTR_Retrotransposon | 11 | 2 | 0.1534 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.3626 |
| Retroviral | 18 | 12 | 0.5970 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 47 unique values; std `0.1470`; degenerate: `False`
- **Class vs Rank gap:** 0.0020 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.2460` (NOT significant vs label-shuffle null; null mean `0.1292`)