# all_inactive

**Pooled OOF CLS:** `0.0000` (PR-AUC `0.3684`, WSpearman `0.0000`)

vs Mandrake baseline `0.318`: **-0.3180**


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

- **Score degeneracy:** 1 unique values; std `0.0000`; degenerate: `True`
- **Class vs Rank gap:** 0.3684 (classifier-only: `False`)
- **Family-constant predictions:** `True` (min within-family std `0.0000`)
- **Permutation p-value:** `1.0000` (NOT significant vs label-shuffle null; null mean `0.1243`)