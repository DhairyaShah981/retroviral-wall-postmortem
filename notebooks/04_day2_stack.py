"""04 — Day 2: structural features + isotonic-calibrated stack + active-learning demo.

Three parallel pushes:
  1. Extract structural features from ESMFold PDBs (12 family-agnostic geometry features)
  2. Stack RF classifier + RF regressor outputs with isotonic calibration
  3. Run the active-learning candidate selector on a held-out family and report
     capture rate vs the trivial "predict top by efficiency" baseline.
"""
from __future__ import annotations
import os, sys, warnings
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
warnings.filterwarnings("ignore")

from mandrake_bench import metrics, cv, audit, report, structural, active

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results")
os.makedirs(RESULTS, exist_ok=True)

# --- Load core data
gt = pd.read_csv(os.path.join(DATA, "rt_sequences.csv")).sort_values("rt_name").reset_index(drop=True)
feats = pd.read_csv(os.path.join(DATA, "handcrafted_features.csv")).set_index("rt_name").loc[gt["rt_name"]].reset_index()

y = gt["active"].values.astype(int)
pe = gt["pe_efficiency_pct"].values.astype(float)
fams = gt["rt_family"].values
rt_names = gt["rt_name"].tolist()

# --- Phase 1: extract structural features (cached)
struct_cache = os.path.join(RESULTS, "structural_features.csv")
if os.path.exists(struct_cache):
    print(f"Loading cached structural features from {struct_cache}")
    struct_df = pd.read_csv(struct_cache).set_index("rt_name")
else:
    print(f"Extracting structural features from 57 PDBs...")
    struct_df = structural.extract_all(os.path.join(DATA, "structures"), rt_names)
    struct_df.to_csv(struct_cache)
    print(f"  cached → {struct_cache}")
print(f"  structural feature shape: {struct_df.shape}")
print(f"  per-feature NaN counts: {struct_df.isna().sum().to_dict()}\n")

# Combine handcrafted + structural
DROP_COLS = {"rt_name", "rt_family", "yxdd_seq"}
hand_cols = [c for c in feats.columns if c not in DROP_COLS]
X_hand = feats[hand_cols].apply(pd.to_numeric, errors="coerce").values
X_struct = struct_df.loc[rt_names].values
X_combined = np.concatenate([X_hand, X_struct], axis=1)
print(f"Handcrafted: {X_hand.shape}, Structural: {X_struct.shape}, Combined: {X_combined.shape}\n")


# --- Phase 2: isotonic-calibrated stack
def isotonic_stack_runner(X_train, y_train, X_test, pe_train=None, **kw):
    """RF clf prob + RF reg prediction → isotonic-calibrate the reg output
    to a [0,1] scale matching the clf's probability scale, then geometric mean.
    """
    pipe_clf = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(n_estimators=400, max_features="sqrt",
                                       min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    pipe_reg = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("rf", RandomForestRegressor(n_estimators=400, max_features="sqrt",
                                      min_samples_leaf=2, random_state=42, n_jobs=-1)),
    ])
    pipe_clf.fit(X_train, y_train)
    pipe_reg.fit(X_train, pe_train if pe_train is not None else y_train.astype(float))

    # Out-of-bag style calibration: get reg train-predictions, fit isotonic to
    # map them onto the active indicator. Use train OOB predictions if available.
    try:
        train_reg = pipe_reg.named_steps["rf"].oob_prediction_  # only if oob_score=True
    except AttributeError:
        train_reg = pipe_reg.predict(X_train)
    iso = IsotonicRegression(out_of_bounds="clip").fit(train_reg, y_train)

    clf_test = pipe_clf.predict_proba(X_test)[:, 1]
    reg_test_raw = pipe_reg.predict(X_test)
    reg_test_iso = iso.transform(reg_test_raw)

    # Geometric mean keeps both contributions multiplicative — if either says 0,
    # final says 0. Add tiny floor so ranking is preserved.
    return np.sqrt(np.clip(clf_test, 1e-6, 1.0) * np.clip(reg_test_iso, 1e-6, 1.0))


# Simple RF baselines on each feature set for comparison
def rf_clf_simple(X_train, y_train, X_test, **kw):
    pipe = Pipeline([("imp", SimpleImputer(strategy="median")),
                     ("rf", RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))])
    pipe.fit(X_train, y_train)
    return pipe.predict_proba(X_test)[:, 1]


runs = {}
runs["rf_simple_handonly"] = cv.lofo_predict(X_hand, y, fams, rf_clf_simple)
runs["rf_simple_combined"] = cv.lofo_predict(X_combined, y, fams, rf_clf_simple)
runs["rf_simple_structonly"] = cv.lofo_predict(X_struct, y, fams, rf_clf_simple)
runs["isotonic_stack_handonly"] = cv.lofo_predict(X_hand, y, fams, isotonic_stack_runner, pe_efficiency=pe)
runs["isotonic_stack_combined"] = cv.lofo_predict(X_combined, y, fams, isotonic_stack_runner, pe_efficiency=pe)

# Score
summary = []
for name, scores in runs.items():
    res = metrics.cls(y, scores, pe)
    pf = metrics.per_family_pr_auc(y, scores, fams)
    rec = {"model": name, "cls": res["cls"], "pr_auc": res["pr_auc"], "w_spearman": res["w_spearman"]}
    for fam in ["Retroviral", "Retron", "LTR_Retrotransposon", "Group_II_Intron"]:
        rec[f"PRAUC_{fam}"] = pf.get(fam, {}).get("pr_auc")
    summary.append(rec)
    a = audit.run_full_audit(y, scores, pe, fams)
    md = report.markdown(name, y, scores, pe, fams, audit_results=a, baseline_cls=0.318)
    with open(os.path.join(RESULTS, f"04_{name}.md"), "w") as f:
        f.write(md)
    pd.DataFrame({"rt_name": gt["rt_name"], "predicted_score": scores}).to_csv(
        os.path.join(RESULTS, f"04_{name}_predictions.csv"), index=False
    )

summary_df = pd.DataFrame(summary).sort_values("cls", ascending=False)
print("=== Day 2 model summary ===")
print(summary_df.to_string(index=False))
summary_df.to_csv(os.path.join(RESULTS, "04_summary.csv"), index=False)

# --- Phase 3: active-learning demo
print("\n=== Active learning capture-rate demo ===")
print("For each held-out family, train on remaining 6 families, run active.next_batch()")
print("on the held-out pool, and report how much of the family's total PE efficiency")
print("we'd have captured by sending top-K to wet-lab.\n")

al_results = []
for held_out_fam in sorted(set(fams)):
    test_mask = fams == held_out_fam
    train_mask = ~test_mask
    n_test = int(test_mask.sum())
    if n_test == 0:
        continue
    k = min(max(1, n_test // 3), 5, n_test)  # send top-k to lab; never exceed pool
    mean_p, std_p = active.rf_predict_with_uncertainty(
        X_combined[train_mask], pe[train_mask], X_combined[test_mask]
    )
    sim = active.regret_simulation(pe[test_mask], mean_p, std_p, k=k)
    # Trivial baseline: random selection. Skip if k == n_test (no choice).
    rng = np.random.default_rng(42)
    if k >= n_test:
        rand_capture = 1.0
    else:
        rand_capture = float(np.mean([
            np.asarray(pe[test_mask])[rng.choice(n_test, size=k, replace=False)].sum() / max(pe[test_mask].sum(), 1e-9)
            for _ in range(200)
        ]))
    al_results.append({
        "held_out_family": held_out_fam,
        "n_in_family": n_test,
        "n_active": int(y[test_mask].sum()),
        "k_sent_to_lab": k,
        "captured_pct": sim["capture_rate"] * 100,
        "random_baseline_pct": rand_capture * 100,
        "lift_pp": (sim["capture_rate"] - rand_capture) * 100,
    })

al_df = pd.DataFrame(al_results)
print(al_df.to_string(index=False))
al_df.to_csv(os.path.join(RESULTS, "04_active_learning_demo.csv"), index=False)
print(f"\nMean capture rate (active): {al_df['captured_pct'].mean():.1f}%")
print(f"Mean capture rate (random): {al_df['random_baseline_pct'].mean():.1f}%")
print(f"Mean lift: +{al_df['lift_pp'].mean():.1f} pp")
