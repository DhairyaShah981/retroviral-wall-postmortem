# lgbm_clf_allfeat

**Pooled OOF CLS:** `0.3035` (PR-AUC `0.3427`, WSpearman `0.2723`)

vs Mandrake baseline `0.318`: **-0.0145**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.2250 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.5490 |
| Retroviral | 18 | 12 | 0.7425 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 51 unique values; std `0.2269`; degenerate: `False`
- **Class vs Rank gap:** 0.0704 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.1780` (NOT significant vs label-shuffle null; null mean `0.1311`)