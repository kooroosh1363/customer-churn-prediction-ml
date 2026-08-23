# Model Card

## Intended use

Educational/portfolio demonstration of a binary churn-risk modeling workflow on IBM's fictional Telco Customer Churn sample.

## Target

`Churn = Yes` means the sample customer left within the last month.

This is a snapshot classification task. The dataset does not provide repeated customer histories, intervention assignment, or a future deployment cohort, so the project does not claim causal retention impact or temporal generalization.

## Models

- Logistic Regression: interpretable linear baseline with balanced class weights.
- Random Forest: non-linear challenger with balanced subsample weighting.

The selected model is the one with the highest validation PR-AUC. The untouched test set is evaluated only after model selection and threshold tuning are complete.

## Data leakage controls

- customer identifiers are removed from predictors
- target is removed before feature construction used by the model
- train/validation/test customer IDs are asserted disjoint
- learned imputers, scaling and one-hot encoding are fitted inside the scikit-learn pipeline on training data only
- validation data selects the model and operating threshold
- test data is reserved for final evaluation

## Decision threshold

The default probability threshold of 0.5 is not assumed to be business-optimal. The repository searches thresholds from 0.10 to 0.90 on the validation set using an explicit scenario cost:

- false negative (missed churner): 500 cost units
- false positive (unnecessary retention action): 50 cost units

These are illustrative assumptions, not observed financial values. The resulting threshold should therefore be interpreted as a scenario analysis rather than a production recommendation. The final report keeps test metrics at both the default 0.5 threshold and the validation-optimized threshold so the trade-off remains visible.

## Metrics

ROC-AUC and PR-AUC evaluate ranking quality. Precision, recall, F1 and the confusion matrix evaluate a chosen operating point. PR-AUC is used for model selection because churn is the minority class.

## Explainability

SHAP is applied to the model actually selected by validation PR-AUC:

- `LinearExplainer` for Logistic Regression
- `TreeExplainer` for Random Forest

The exported mean absolute SHAP values are global feature-importance diagnostics on held-out test examples. They are not causal effects.

## Error analysis

Subgroup precision and recall are exported for `Contract` and `InternetService`. These are diagnostic slices for understanding model behavior and do not establish formal fairness or absence of disparate impact.

## Validated run snapshot

In the current CI-validated run, Logistic Regression won on validation PR-AUC. The final test ranking metrics were approximately ROC-AUC 0.832 and PR-AUC 0.623. Under the illustrative 10:1 false-negative/false-positive cost ratio, the validation-selected operating threshold was approximately 0.20, producing very high recall with lower precision. These numbers describe this reproducible sample run only.

## Limitations

- fictional public sample, not production telecom data
- one historical snapshot; no temporal backtest
- no intervention/treatment data, so the model predicts churn risk rather than uplift
- scenario costs are assumptions rather than measured retention economics
- no probability-calibration study is claimed
- no external validation dataset is used
- subgroup diagnostics are not a formal fairness audit
