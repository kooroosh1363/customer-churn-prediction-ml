# DS-02 — Customer Churn Prediction

Portfolio-grade supervised ML system for predicting telecom customer churn from the public IBM Telco Customer Churn sample.

## What this project demonstrates

- binary classification with a leakage-aware train/validation/test split
- reproducible feature engineering with scikit-learn pipelines
- class-imbalance handling via class weights
- baseline vs. non-linear model comparison
- ROC-AUC, PR-AUC, precision, recall, F1 and confusion-matrix evaluation
- validation-only decision-threshold tuning using an explicit business-cost scenario
- SHAP explainability for the selected tree model
- error analysis and subgroup metrics
- reproducible data acquisition, tests and GitHub Actions CI

## Data

The project uses IBM's fictional Telco Customer Churn sample: 7,043 customers and 21 source columns. `Churn` indicates whether a customer left within the last month. The raw CSV is downloaded at runtime and excluded from Git. See [DATA_SOURCE.md](DATA_SOURCE.md).

## Important claim boundary

The dataset is a public fictional sample. Model metrics demonstrate methodology on this sample; they are not production churn estimates. Business costs used for threshold tuning are documented scenario assumptions, not observed company economics.

## Architecture

```text
IBM public CSV
    -> download + schema validation
    -> leakage-safe feature engineering
    -> stratified train / validation / test split
    -> logistic baseline + random-forest challenger
    -> validation threshold optimization
    -> untouched test evaluation
    -> SHAP + subgroup error analysis
    -> JSON/CSV artifacts
    -> pytest + GitHub Actions CI
```

## Run locally

```bash
python -m pip install -r requirements.txt
python -m src.train
pytest -q
```

Training writes reproducible outputs into `artifacts/` (ignored by Git): metrics, threshold search, subgroup metrics, feature importance, and the fitted model.

## Interview signal

This project is deliberately different from descriptive churn analytics. It focuses on the predictive ML lifecycle: target definition, leakage prevention, imbalance, model selection, threshold decisions, explainability, error analysis, and reproducibility.
