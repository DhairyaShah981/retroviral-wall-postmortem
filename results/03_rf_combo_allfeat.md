# rf_combo_allfeat

**Pooled OOF CLS:** `0.3178` (PR-AUC `0.4207`, WSpearman `0.2553`)

vs Mandrake baseline `0.318`: **-0.0002**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.2429 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.5052 |
| Retroviral | 18 | 12 | 0.7024 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 54 unique values; std `0.2085`; degenerate: `False`
- **Class vs Rank gap:** 0.1654 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.1360` (NOT significant vs label-shuffle null; null mean `0.1196`)