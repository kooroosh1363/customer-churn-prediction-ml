# DS-02 — Customer Churn Prediction

Portfolio-grade supervised ML system for predicting telecom customer churn from the public IBM Telco Customer Churn sample.

## What this project demonstrates

- binary classification with a leakage-aware train/validation/test split
- reproducible feature engineering with scikit-learn pipelines
- class-imbalance handling via class weights
- logistic-regression baseline vs. random-forest challenger
- ROC-AUC, PR-AUC, precision, recall, F1 and confusion-matrix evaluation
- model selection on validation PR-AUC
- validation-only decision-threshold tuning using an explicit business-cost scenario
- untouched test evaluation at both 0.5 and the optimized threshold
- SHAP explainability for the model that is actually selected
- subgroup error analysis for Contract and InternetService
- reproducible data acquisition, tests and GitHub Actions CI

## Data

The project uses IBM's fictional Telco Customer Churn sample: 7,043 customers and 21 source columns. `Churn` indicates whether a customer left within the last month. The raw CSV is downloaded at runtime and excluded from Git. See [DATA_SOURCE.md](DATA_SOURCE.md).

## Important claim boundary

The dataset is a public fictional sample. Model metrics demonstrate methodology on this sample; they are not production churn estimates. Business costs used for threshold tuning are documented scenario assumptions, not observed company economics. Because this is one historical snapshot, this repository does not claim temporal generalization, causal retention impact, or uplift-model performance.

## Architecture

```text
IBM public CSV
    -> download + schema validation
    -> deterministic feature engineering
    -> stratified train / validation / test split
    -> pipeline-fitted preprocessing on training data only
    -> logistic baseline + random-forest challenger
    -> select model by validation PR-AUC
    -> optimize threshold on validation only
    -> untouched test evaluation at 0.5 and optimized threshold
    -> SHAP for selected model + subgroup error analysis
    -> JSON/CSV artifacts
    -> pytest + GitHub Actions CI
```

## Current validated run

The CI-validated run selected Logistic Regression on validation PR-AUC.

- validation PR-AUC: about 0.676
- test ROC-AUC: about 0.832
- test PR-AUC: about 0.623
- scenario-optimized threshold: about 0.20
- optimized-threshold recall: about 0.957

The low operating threshold is a consequence of the documented illustrative cost ratio where a missed churner is treated as 10× more costly than an unnecessary retention action. It should not be interpreted as a universally optimal churn threshold.

## Run locally

```bash
python -m pip install -r requirements.txt
python -m src.train
python -m pytest -q
```

Training writes reproducible outputs into `artifacts/` (ignored by Git): metrics, threshold search, subgroup metrics, selected-model SHAP feature importance, and the fitted model.

## Interview signal

This project is deliberately different from descriptive churn analytics. It focuses on the predictive ML lifecycle: target definition, leakage prevention, imbalance, baseline/challenger comparison, model selection, threshold decisions, explainability, error analysis, claim boundaries, and reproducibility.
