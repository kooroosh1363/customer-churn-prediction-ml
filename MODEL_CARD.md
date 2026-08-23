# Model Card

## Intended use

Educational/portfolio demonstration of a binary churn-risk modeling workflow on IBM's fictional Telco Customer Churn sample.

## Target

`Churn = Yes` means the sample customer left within the last month.

## Models

- Logistic Regression: interpretable linear baseline with balanced class weights.
- Random Forest: non-linear challenger with balanced subsample weighting.

The selected model is the one with the highest validation PR-AUC. The untouched test set is evaluated only after model selection and threshold tuning.

## Decision threshold

The default probability threshold of 0.5 is not assumed to be business-optimal. The repository searches thresholds from 0.10 to 0.90 on the validation set using an explicit scenario cost:

- false negative (missed churner): 500 cost units
- false positive (unnecessary retention action): 50 cost units

These are illustrative assumptions, not observed financial values.

## Metrics

ROC-AUC and PR-AUC evaluate ranking quality. Precision, recall, F1 and the confusion matrix evaluate the chosen operating point. PR-AUC is emphasized because churn is the minority class.

## Explainability

SHAP TreeExplainer is applied to the fitted Random Forest challenger and exports mean absolute SHAP importance on a held-out test sample. This is global feature importance, not a causal interpretation.

## Limitations

- fictional public sample, not production telecom data
- one historical snapshot; no temporal backtest
- no intervention/treatment data, so the model predicts churn risk rather than uplift
- no claim that the scenario costs reflect real retention economics
- subgroup metrics are diagnostic and do not establish fairness or absence of disparate impact
