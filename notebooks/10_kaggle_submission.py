"""10 — Generate a Kaggle-format submission CSV.

The example submission has TWO columns: predicted_active (binary) AND
predicted_score (continuous). The Kaggle public metric appears to score on
one of these (likely binary accuracy or PR-AUC on predicted_score), which is
why 6 teams tied at 1.00000 — they predicted the binary ground truth.

The official Mandrake evaluator scores on predicted_score with CLS.

This script writes a properly-formatted submission of our best model
(isotonic_stack_handonly, CLS 0.5452 per official evaluator), with:
  - predicted_active = (predicted_score > optimal_threshold)
  - predicted_score  = the isotonic stack score (raw, not binary)

Threshold is chosen to maximize F1 on the full LOFO predictions, mirroring
what a real wet-lab triage decision would optimize.

Once Kaggle creds are set up (~/.kaggle/kaggle.json), submit with:
    kaggle competitions submit -c retroviral-challenge-predict \\
        -f results/10_kaggle_submission.csv -m "Isotonic stack, CLS 0.5452"
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")

gt = pd.read_csv(os.path.join(DATA, "rt_sequences.csv")).sort_values("rt_name").reset_index(drop=True)
preds = pd.read_csv(os.path.join(RESULTS, "04_isotonic_stack_handonly_predictions.csv"))
merged = gt.merge(preds, on="rt_name")
y = merged["active"].values
s = merged["predicted_score"].values

# Find optimal F1 threshold on LOFO OOF (proxy for what'd work on Stage 2)
best_t, best_f1 = 0.5, 0.0
for t in np.linspace(s.min(), s.max(), 200):
    f = f1_score(y, (s > t).astype(int))
    if f > best_f1:
        best_t, best_f1 = float(t), float(f)
print(f"Best F1 threshold on LOFO OOF: {best_t:.4f} (F1 = {best_f1:.3f})")

submission = pd.DataFrame({
    "rt_name": merged["rt_name"],
    "predicted_active": (s > best_t).astype(int),
    "predicted_score": s,
})
out = os.path.join(RESULTS, "10_kaggle_submission.csv")
submission.to_csv(out, index=False)
print(f"\nSubmission written → {out}")
print(submission.head().to_string(index=False))
print(f"\n{submission['predicted_active'].sum()} predicted active / {len(submission)} total")
print(f"True actives: {y.sum()} / {len(y)}")
print(f"\nPredicted-active confusion vs truth:")
print(pd.crosstab(submission['predicted_active'], y, rownames=['pred'], colnames=['true']))
