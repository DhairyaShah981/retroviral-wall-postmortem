"""Reference baselines for sanity-checking new models."""
from __future__ import annotations
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


def predict_all_inactive(X_train, y_train, X_test, **kw):
    return np.zeros(len(X_test))


def predict_random(X_train, y_train, X_test, seed=0, **kw):
    rng = np.random.default_rng(seed)
    return rng.uniform(size=len(X_test))


def predict_family_prior(X_train, y_train, X_test, train_families=None, test_families=None, **kw):
    """Constant per-test-family prediction = train base-rate of active across all train families.
    This is a deliberately weak baseline showing what 'no signal' looks like.
    """
    prior = float(np.mean(y_train)) if len(y_train) else 0.0
    return np.full(len(X_test), prior)


def make_rf_classifier(n_estimators=300, random_state=42):
    """Vanilla RF classifier — Mandrake's reference baseline (handcrafted + RF → CLS 0.318)."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("rf", RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=-1)),
    ])


def predict_rf_classifier(X_train, y_train, X_test, **kw):
    clf = make_rf_classifier()
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]


def predict_rf_regressor(X_train, y_train, X_test, pe_train=None, **kw):
    """Regress directly on pe_efficiency_pct. Better aligned with WSpearman."""
    if pe_train is None:
        raise ValueError("RF regressor needs pe_train (efficiency targets).")
    reg = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ])
    reg.fit(X_train, pe_train)
    return reg.predict(X_test)


def predict_esm_logreg(X_train, y_train, X_test, **kw):
    """ESM-only logistic regression — known to overfit family. Useful as a 'family-leak' canary."""
    clf = Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(C=0.1, max_iter=2000, random_state=42)),
    ])
    clf.fit(X_train, y_train)
    return clf.predict_proba(X_test)[:, 1]
