# isotonic_handonly_plus_id

**Pooled OOF CLS:** `0.5105` (PR-AUC `0.5169`, WSpearman `0.5042`)

vs Mandrake baseline `0.318`: **+0.1925**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.1944 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.4556 |
| Retroviral | 18 | 12 | 0.7897 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 45 unique values; std `0.2544`; degenerate: `False`
- **Class vs Rank gap:** 0.0127 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0036` (SIGNIFICANT vs label-shuffle null; null mean `0.1220`)