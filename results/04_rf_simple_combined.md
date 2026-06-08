# rf_simple_combined

**Pooled OOF CLS:** `0.4300` (PR-AUC `0.3669`, WSpearman `0.5194`)

vs Mandrake baseline `0.318`: **+0.1120**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 1.0000 |
| LTR_Retrotransposon | 11 | 2 | 0.1625 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.6242 |
| Retroviral | 18 | 12 | 0.7869 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 49 unique values; std `0.1334`; degenerate: `False`
- **Class vs Rank gap:** 0.1525 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.0382` (SIGNIFICANT vs label-shuffle null; null mean `0.1227`)