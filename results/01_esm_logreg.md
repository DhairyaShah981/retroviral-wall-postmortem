# esm_logreg

**Pooled OOF CLS:** `0.1837` (PR-AUC `0.3811`, WSpearman `0.1210`)

vs Mandrake baseline `0.318`: **-0.1343**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.8333 |
| LTR_Retrotransposon | 11 | 2 | 0.4500 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.7862 |
| Retroviral | 18 | 12 | 0.8123 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 53 unique values; std `0.2166`; degenerate: `False`
- **Class vs Rank gap:** 0.2601 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.3160` (NOT significant vs label-shuffle null; null mean `0.1142`)