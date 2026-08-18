"""
Local Streamlit workbench for QSAR biodegradability classifiers.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

APP_DIR = Path(__file__).resolve().parent
STORE_DIR = APP_DIR / "model"
HOLDOUT_CSV = APP_DIR / "test_data.csv"
SUMMARY_FILE = STORE_DIR / "eval_summary.json"

PAGE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=Space+Grotesk:wght@500;700&display=swap');
html, body, [class*="css"]  {
  font-family: 'IBM Plex Sans', sans-serif;
}
.hero-band {
  background: linear-gradient(135deg, #0b3d2e 0%, #1f6f54 48%, #8fb339 100%);
  color: #f4fff8;
  padding: 1.4rem 1.6rem;
  border-radius: 0;
  margin-bottom: 1.2rem;
}
.hero-band h1 {
  font-family: 'Space Grotesk', sans-serif;
  font-size: 2rem;
  margin: 0 0 0.35rem 0;
}
.hero-band p { margin: 0; opacity: 0.92; }
.metric-strip {
  display: grid;
  grid-template-columns: repeat(6, minmax(0, 1fr));
  gap: 0.55rem;
  margin: 0.8rem 0 1.1rem 0;
}
.metric-cell {
  background: #f2f7f4;
  border-left: 4px solid #1f6f54;
  padding: 0.65rem 0.7rem;
}
.metric-cell span { display:block; font-size: 0.75rem; color: #40574c; }
.metric-cell strong { font-size: 1.15rem; color: #0b3d2e; }
section[data-testid="stSidebar"] {
  background: #eef6f1;
}
</style>
"""


@st.cache_resource
def read_summary() -> dict:
    return json.loads(SUMMARY_FILE.read_text(encoding="utf-8"))


@st.cache_resource
def fetch_estimator(display_name: str):
    summary = read_summary()
    file_name = summary["artifact_map"][display_name]
    return joblib.load(STORE_DIR / file_name)


def split_features_and_label(
    frame: pd.DataFrame, column_order: list[str], label_key: str
):
    absent = [col for col in column_order if col not in frame.columns]
    if absent:
        preview = ", ".join(absent[:6])
        suffix = " ..." if len(absent) > 6 else ""
        raise ValueError(f"CSV is missing descriptor columns: {preview}{suffix}")
    if label_key not in frame.columns:
        raise ValueError(f"CSV must include label column '{label_key}'.")

    feature_block = frame[column_order].apply(pd.to_numeric, errors="coerce")
    label_series = pd.to_numeric(frame[label_key], errors="coerce")
    valid_rows = feature_block.notna().all(axis=1) & label_series.notna()
    feature_block = feature_block.loc[valid_rows]
    label_series = label_series.loc[valid_rows].astype(int)
    if feature_block.empty:
        raise ValueError("No usable rows after numeric cleaning.")
    return feature_block, label_series


def evaluate_predictions(actual, predicted, positive_scores) -> dict[str, float]:
    return {
        "Accuracy": accuracy_score(actual, predicted),
        "AUC": roc_auc_score(actual, positive_scores),
        "Precision": precision_score(actual, predicted, zero_division=0),
        "Recall": recall_score(actual, predicted, zero_division=0),
        "F1": f1_score(actual, predicted, zero_division=0),
        "MCC": matthews_corrcoef(actual, predicted),
    }


def draw_confusion(actual, predicted, tick_labels: list[str]):
    matrix = confusion_matrix(actual, predicted)
    figure, axis = plt.subplots(figsize=(5.4, 4.3))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="YlGn",
        xticklabels=tick_labels,
        yticklabels=tick_labels,
        ax=axis,
        cbar=True,
    )
    axis.set_xlabel("Predicted class")
    axis.set_ylabel("True class")
    axis.set_title("Confusion matrix")
    figure.tight_layout()
    return figure


def render_metric_strip(values: dict[str, float]) -> None:
    cells = []
    for title, number in values.items():
        cells.append(
            f'<div class="metric-cell"><span>{title}</span><strong>{number:.4f}</strong></div>'
        )
    st.markdown(
        f'<div class="metric-strip">{"".join(cells)}</div>',
        unsafe_allow_html=True,
    )


def launch() -> None:
    st.set_page_config(
        page_title="QSAR Biodegradability Workbench",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    summary = read_summary()
    column_order = summary["feature_names"]
    label_key = summary["target_column"]
    class_map = summary["class_names"]
    tick_labels = [class_map[str(i)] for i in sorted(int(k) for k in class_map)]
    estimator_names = list(summary["artifact_map"].keys())

    st.markdown(
        """
        <div class="hero-band">
          <h1>QSAR Biodegradability Workbench</h1>
          <p>Compare five classical classifiers on molecular descriptors predicting ready biodegradation.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown("### Experiment controls")
        uploaded_csv = st.file_uploader(
            "Hold-out CSV (descriptors + biodeg_label)",
            type=["csv"],
            help="Prefer the bundled test_data.csv. Keep only evaluation rows.",
        )
        chosen_model = st.selectbox("Classifier", estimator_names)
        st.markdown("---")
        st.write(f"Dataset: {summary['dataset']}")
        st.write(f"Descriptors: {summary['n_features']}")
        st.write(f"Full sample size: {summary['n_instances']}")
        st.write(f"Training seed: {summary['seed']}")

    if uploaded_csv is not None:
        working_frame = pd.read_csv(uploaded_csv)
        origin_note = "uploaded hold-out file"
    else:
        working_frame = pd.read_csv(HOLDOUT_CSV)
        origin_note = "bundled test_data.csv"

    st.markdown("#### Hold-out preview")
    st.caption(f"Source: {origin_note} | shape={working_frame.shape}")
    st.dataframe(working_frame.head(12), use_container_width=True)

    try:
        feature_block, label_series = split_features_and_label(
            working_frame, column_order, label_key
        )
    except ValueError as problem:
        st.error(str(problem))
        st.stop()

    estimator = fetch_estimator(chosen_model)
    predicted = estimator.predict(feature_block)
    positive_scores = estimator.predict_proba(feature_block)[:, 1]
    live_scores = evaluate_predictions(label_series, predicted, positive_scores)

    st.markdown(f"#### Live scores — {chosen_model}")
    render_metric_strip(live_scores)

    matrix_col, report_col = st.columns([1.05, 1])
    with matrix_col:
        st.pyplot(draw_confusion(label_series, predicted, tick_labels), clear_figure=True)
    with report_col:
        st.markdown("##### Classification report")
        text_report = classification_report(
            label_series,
            predicted,
            target_names=tick_labels,
            digits=4,
            zero_division=0,
        )
        st.code(text_report, language="text")

    st.markdown("#### Training-script scoreboard (fixed hold-out)")
    board = (
        pd.DataFrame(summary["metrics"])
        .T.reset_index()
        .rename(columns={"index": "ML Model Name"})
    )
    metric_cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
    board[metric_cols] = board[metric_cols].astype(float).round(4)
    st.dataframe(board, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    launch()
