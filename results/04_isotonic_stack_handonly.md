# isotonic_stack_handonly

**Pooled OOF CLS:** `0.5442` (PR-AUC `0.6003`, WSpearman `0.4976`)

vs Mandrake baseline `0.318`: **+0.2262**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.2679 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.5852 |
| Retroviral | 18 | 12 | 0.8269 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 47 unique values; std `0.2466`; degenerate: `False`
- **Class vs Rank gap:** 0.1028 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0016` (SIGNIFICANT vs label-shuffle null; null mean `0.1221`)