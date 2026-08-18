"""
Build and persist classifiers for Assignment 2 (QSAR biodegradability).

Binary task: ready biodegradation (RB) vs not ready (NRB).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = Path(__file__).resolve().parent
SEED = 2026
HOLD_OUT_RATIO = 0.25

# Display names keep the assignment table labels; artifact files use distinct ids.
ESTIMATOR_BUNDLE = {
    "Logistic Regression": "artifact_logreg.joblib",
    "Decision Tree": "artifact_dtree.joblib",
    "kNN": "artifact_knn.joblib",
    "Naive Bayes": "artifact_gauss_nb.joblib",
    "Random Forest (Ensemble)": "artifact_rf_ensemble.joblib",
}


def pull_qsar_frame() -> tuple[pd.DataFrame, pd.Series, dict[str, str]]:
    bundle = fetch_openml(name="qsar-biodeg", version=1, as_frame=True, parser="auto")
    predictors = bundle.data.copy()
    predictors.columns = [f"mol_desc_{idx:02d}" for idx in range(1, predictors.shape[1] + 1)]

    encoder = LabelEncoder()
    labels = pd.Series(encoder.fit_transform(bundle.target), name="biodeg_label")
    label_lookup = {
        str(code): str(cls_name) for code, cls_name in enumerate(encoder.classes_)
    }
    return predictors, labels, label_lookup


def score_holdout(actual, predicted, positive_scores) -> dict[str, float]:
    return {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "AUC": float(roc_auc_score(actual, positive_scores)),
        "Precision": float(precision_score(actual, predicted, zero_division=0)),
        "Recall": float(recall_score(actual, predicted, zero_division=0)),
        "F1": float(f1_score(actual, predicted, zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predicted)),
    }


def assemble_estimators() -> dict[str, object]:
    """Hyperparameters chosen for this QSAR setup (not default textbook stubs)."""
    return {
        "Logistic Regression": Pipeline(
            steps=[
                ("zscore", StandardScaler()),
                (
                    "estimator",
                    LogisticRegression(
                        C=0.65,
                        solver="saga",
                        max_iter=4500,
                        class_weight="balanced",
                        random_state=SEED,
                    ),
                ),
            ]
        ),
        "Decision Tree": DecisionTreeClassifier(
            criterion="entropy",
            max_depth=9,
            min_samples_split=14,
            min_samples_leaf=6,
            class_weight="balanced",
            random_state=SEED,
        ),
        "kNN": Pipeline(
            steps=[
                ("zscore", StandardScaler()),
                (
                    "estimator",
                    KNeighborsClassifier(
                        n_neighbors=11,
                        weights="uniform",
                        metric="manhattan",
                        p=1,
                    ),
                ),
            ]
        ),
        "Naive Bayes": GaussianNB(var_smoothing=2.5e-8),
        "Random Forest (Ensemble)": RandomForestClassifier(
            n_estimators=275,
            max_depth=14,
            min_samples_split=8,
            min_samples_leaf=3,
            max_features="sqrt",
            bootstrap=True,
            class_weight="balanced_subsample",
            random_state=SEED,
            n_jobs=-1,
        ),
    }


def run_training() -> None:
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    predictors, labels, label_lookup = pull_qsar_frame()

    train_x, test_x, train_y, test_y = train_test_split(
        predictors,
        labels,
        test_size=HOLD_OUT_RATIO,
        random_state=SEED,
        stratify=labels,
    )

    holdout_table = test_x.copy()
    holdout_table["biodeg_label"] = test_y.to_numpy()
    holdout_csv = PROJECT_ROOT / "test_data.csv"
    holdout_table.to_csv(holdout_csv, index=False)

    scoreboard: dict[str, dict[str, float]] = {}
    for display_name, estimator in assemble_estimators().items():
        estimator.fit(train_x, train_y)
        predicted = estimator.predict(test_x)
        positive_scores = estimator.predict_proba(test_x)[:, 1]
        scoreboard[display_name] = score_holdout(test_y, predicted, positive_scores)
        joblib.dump(estimator, ARTIFACT_DIR / ESTIMATOR_BUNDLE[display_name])
        print(f"{display_name}: {scoreboard[display_name]}")

    summary = {
        "dataset": "UCI QSAR Biodegradation (OpenML: qsar-biodeg v1)",
        "task": "Binary classification of chemical biodegradability (RB vs NRB)",
        "n_features": int(predictors.shape[1]),
        "n_instances": int(predictors.shape[0]),
        "feature_names": list(predictors.columns),
        "target_column": "biodeg_label",
        "class_names": label_lookup,
        "hold_out_ratio": HOLD_OUT_RATIO,
        "seed": SEED,
        "artifact_map": ESTIMATOR_BUNDLE,
        "metrics": scoreboard,
    }
    summary_path = ARTIFACT_DIR / "eval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"\nHold-out CSV -> {holdout_csv}")
    print(f"Eval summary -> {summary_path}")


if __name__ == "__main__":
    run_training()
