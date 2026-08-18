# ML Assignment 2 — QSAR Biodegradability Classification

End-to-end classification workbench on the UCI QSAR Biodegradation dataset.  
Streamlit Community Cloud deployment is not part of this submission.

## Local usage

```bash
cd asignment2
pip install -r requirements.txt
python model/fit_classifiers.py
streamlit run app.py
```

## Layout

```text
asignment2/
|-- app.py
|-- requirements.txt
|-- README.md
|-- test_data.csv
|-- model/
    |-- fit_classifiers.py
    |-- eval_summary.json
    |-- artifact_*.joblib
```

---

## a. Problem statement

Chemical compounds are labeled as ready biodegradable (RB) or not ready biodegradable (NRB) using molecular descriptors. This assignment trains multiple classical classifiers on one public dataset, evaluates them with a fixed hold-out split, and exposes the trained estimators through a Streamlit UI for CSV-based inspection.

Implemented estimators:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)

Reported metrics for each estimator: Accuracy, AUC, Precision, Recall, F1, MCC.

---

## b. Dataset description

| Item | Detail |
|------|--------|
| Name | UCI QSAR Biodegradation (OpenML: `qsar-biodeg`, version 1) |
| Source | UCI Machine Learning Repository via OpenML |
| Task | Binary classification — **RB** (ready biodegradable) vs **NRB** (not ready) |
| Instances | 1,055 (≥ 500 required) |
| Features | 41 molecular descriptors (≥ 12 required), renamed locally as `mol_desc_01` … `mol_desc_41` |
| Target | `biodeg_label` (integer-encoded class) |
| Split | Stratified hold-out 25% (`seed=2026`) |
| Test file | `test_data.csv` used by the Streamlit workbench |

Descriptor families include constitutional indices, topological indices, and charge-related molecular attributes used in QSAR modeling.

---

## c. Repository Link

**Repository:** [https://github.com/2025ac05291/ML_ASSIGNMENTS/tree/main/asignment2](https://github.com/2025ac05291/ML_ASSIGNMENTS/tree/main/asignment2)

---

## d. Models used

Metrics are measured on the fixed hold-out set written to `test_data.csv`.

### Comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.8371 | 0.8964 | 0.7255 | 0.8315 | 0.7749 | 0.6519 |
| Decision Tree | 0.7689 | 0.8147 | 0.6373 | 0.7303 | 0.6806 | 0.5038 |
| kNN | 0.8258 | 0.8900 | 0.7312 | 0.7640 | 0.7473 | 0.6147 |
| Naive Bayes | 0.6818 | 0.8531 | 0.5161 | 0.8989 | 0.6557 | 0.4516 |
| Random Forest (Ensemble) | 0.8523 | 0.8992 | 0.7841 | 0.7753 | 0.7797 | 0.6686 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong linear baseline after z-score scaling and class balancing. Accuracy ~0.84 with balanced precision/recall shows molecular descriptors separate fairly linearly once standardized. |
| Decision Tree | Weakest tree-based result here. Entropy splits with depth/leaf constraints reduce memorization but also lose interaction structure present in the descriptors, so both precision and recall lag the linear and ensemble models. |
| kNN | Competitive after scaling when using Manhattan distance and k=11. Neighborhood voting tracks Logistic Regression closely on AUC, yet absolute accuracy remains below Random Forest because sparse descriptor neighborhoods are noisy. |
| Naive Bayes | Highest recall (~0.90) but lowest precision/accuracy. The Gaussian independence assumption is unrealistic for correlated QSAR descriptors, so many NRB compounds are falsely labeled RB. |
| Random Forest (Ensemble) | Best overall trade-off: top Accuracy, AUC, Precision, F1, and MCC. Subsample balancing plus bagged trees capture non-linear descriptor interactions better than a single Decision Tree. |
| **Overall Winner for your dataset?** | **Random Forest (Ensemble)** — leads on five of six metrics and is the most reliable biodegradability detector on this hold-out split. |

---

## Streamlit workbench (local only)

`app.py` provides:

1. CSV upload for hold-out evaluation  
2. Classifier dropdown  
3. Live Accuracy / AUC / Precision / Recall / F1 / MCC  
4. Confusion matrix and classification report  

Community Cloud hosting is intentionally omitted.
