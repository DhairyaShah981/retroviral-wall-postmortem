# isotonic_id_only

**Pooled OOF CLS:** `0.0201` (PR-AUC `0.4438`, WSpearman `0.0103`)

vs Mandrake baseline `0.318`: **-0.2979**


## Per-family PR-AUC

| Family | n | Active | PR-AUC |
|---|---|---|---|
| CRISPR-associated | 5 | 0 | N/A |
| Group_II_Intron | 5 | 2 | 0.7500 |
| LTR_Retrotransposon | 11 | 2 | 0.1742 |
| Other | 5 | 0 | N/A |
| Retron | 12 | 5 | 0.6857 |
| Retroviral | 18 | 12 | 0.8128 |
| Unclassified | 1 | 0 | N/A |

## Audit

- **Score degeneracy:** 39 unique values; std `0.0833`; degenerate: `False`
- **Class vs Rank gap:** 0.4335 (classifier-only: `False`)
- **Family-constant predictions:** `False` (min within-family std `0.0000`)
- **Permutation p-value:** `0.4921` (NOT significant vs label-shuffle null; null mean `0.1224`)