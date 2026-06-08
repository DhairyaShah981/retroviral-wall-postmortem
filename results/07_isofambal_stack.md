# isofambal_stack

**Pooled OOF CLS:** `0.4975` (PR-AUC `0.6369`, WSpearman `0.4082`)

vs Mandrake baseline `0.318`: **+0.1795**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.2083 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.6587 |
| Retroviral | 18 | 12 | 0.8051 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 49 unique values; std `0.2249`; degenerate: `False`
- **Class vs Rank gap:** 0.2287 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0050` (SIGNIFICANT vs label-shuffle null; null mean `0.1227`)