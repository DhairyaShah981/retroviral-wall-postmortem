# random

**Pooled OOF CLS:** `0.3127` (PR-AUC `0.5707`, WSpearman `0.2153`)

vs Mandrake baseline `0.318`: **-0.0053**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.5000 |
| LTR_Retrotransposon | 11 | 2 | 0.7000 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.8242 |
| Retroviral | 18 | 12 | 0.7095 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 18 unique values; std `0.3353`; degenerate: `False`
- **Class vs Rank gap:** 0.3554 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.1400` (NOT significant vs label-shuffle null; null mean `0.1090`)