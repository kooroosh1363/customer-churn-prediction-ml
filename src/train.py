from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
import shap
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score,
    f1_score, confusion_matrix
)
from sklearn.model_selection import train_test_split

from .data import load_data, engineer_features

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RANDOM_STATE = 42
FALSE_NEGATIVE_COST = 500.0
FALSE_POSITIVE_COST = 50.0


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    return ColumnTransformer([
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scale", StandardScaler())
        ]), numeric),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), categorical),
    ], verbose_feature_names_out=False)


def evaluate(y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "threshold": float(threshold),
        "roc_auc": float(roc_auc_score(y_true, prob)),
        "pr_auc": float(average_precision_score(y_true, prob)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "scenario_cost": float(fn * FALSE_NEGATIVE_COST + fp * FALSE_POSITIVE_COST),
    }


def choose_threshold(y_true, prob) -> tuple[float, pd.DataFrame]:
    rows = [evaluate(y_true, prob, t) for t in np.arange(0.10, 0.91, 0.01)]
    table = pd.DataFrame(rows).sort_values(["scenario_cost", "threshold"])
    return float(table.iloc[0]["threshold"]), table.sort_values("threshold")


def subgroup_metrics(X: pd.DataFrame, y: pd.Series, prob: np.ndarray, threshold: float) -> pd.DataFrame:
    pred = (prob >= threshold).astype(int)
    work = X[["Contract", "InternetService"]].copy()
    work["y"] = y.to_numpy()
    work["pred"] = pred
    rows = []
    for col in ["Contract", "InternetService"]:
        for value, g in work.groupby(col, dropna=False):
            rows.append({
                "feature": col,
                "group": str(value),
                "n": len(g),
                "churn_rate": g["y"].mean(),
                "precision": precision_score(g["y"], g["pred"], zero_division=0),
                "recall": recall_score(g["y"], g["pred"], zero_division=0),
            })
    return pd.DataFrame(rows)


def main() -> None:
    ART.mkdir(exist_ok=True)
    df = engineer_features(load_data())
    ids = df.pop("customerID")
    y = (df.pop("Churn") == "Yes").astype(int)
    X_train, X_temp, y_train, y_temp, id_train, id_temp = train_test_split(
        df, y, ids, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test, id_val, id_test = train_test_split(
        X_temp, y_temp, id_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )
    assert set(id_train).isdisjoint(id_val)
    assert set(id_train).isdisjoint(id_test)
    assert set(id_val).isdisjoint(id_test)

    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=4, class_weight="balanced_subsample",
            random_state=RANDOM_STATE, n_jobs=-1
        ),
    }
    fitted = {}
    validation = {}
    for name, estimator in models.items():
        pipe = Pipeline([("prep", build_preprocessor(X_train)), ("model", estimator)])
        pipe.fit(X_train, y_train)
        prob = pipe.predict_proba(X_val)[:, 1]
        validation[name] = evaluate(y_val, prob, 0.5)
        fitted[name] = pipe

    selected_name = max(validation, key=lambda n: validation[n]["pr_auc"])
    selected = fitted[selected_name]
    val_prob = selected.predict_proba(X_val)[:, 1]
    threshold, threshold_table = choose_threshold(y_val, val_prob)
    test_prob = selected.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob, threshold)

    report = {
        "split_sizes": {"train": len(X_train), "validation": len(X_val), "test": len(X_test)},
        "target_rates": {"train": y_train.mean(), "validation": y_val.mean(), "test": y_test.mean()},
        "validation_at_0_5": validation,
        "selected_model": selected_name,
        "threshold_policy": {
            "selected_on": "validation only",
            "false_negative_cost": FALSE_NEGATIVE_COST,
            "false_positive_cost": FALSE_POSITIVE_COST,
            "chosen_threshold": threshold,
        },
        "test_metrics": test_metrics,
    }
    (ART / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    threshold_table.to_csv(ART / "threshold_search.csv", index=False)
    subgroup_metrics(X_test, y_test, test_prob, threshold).to_csv(ART / "subgroup_metrics.csv", index=False)
    joblib.dump(selected, ART / "model.joblib")

    # Explain the fitted random-forest challenger when available, independent of selection.
    rf = fitted["random_forest"]
    transformed = rf.named_steps["prep"].transform(X_test.iloc[:250])
    feature_names = rf.named_steps["prep"].get_feature_names_out()
    explainer = shap.TreeExplainer(rf.named_steps["model"])
    sv = explainer.shap_values(transformed)
    values = sv[1] if isinstance(sv, list) else (sv[:, :, 1] if getattr(sv, "ndim", 0) == 3 else sv)
    importance = np.abs(values).mean(axis=0)
    pd.DataFrame({"feature": feature_names, "mean_abs_shap": importance}).sort_values(
        "mean_abs_shap", ascending=False
    ).to_csv(ART / "shap_feature_importance.csv", index=False)

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
