# isotonic_stack_combined

**Pooled OOF CLS:** `0.4525` (PR-AUC `0.4590`, WSpearman `0.4462`)

vs Mandrake baseline `0.318`: **+0.1345**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.2250 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.6052 |
| Retroviral | 18 | 12 | 0.7078 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 48 unique values; std `0.2170`; degenerate: `False`
- **Class vs Rank gap:** 0.0129 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0280` (SIGNIFICANT vs label-shuffle null; null mean `0.1270`)