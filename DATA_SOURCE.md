# Data Source & Provenance

This project uses IBM's **Telco Customer Churn** sample data. IBM documents the sample as a fictional telecommunications customer dataset in which the churn field indicates whether a customer left within the last month. The sample includes customer demographics, services, tenure, contract/payment information, and charges.

- Source organization: IBM
- Sample: Telco Customer Churn
- Rows: 7,043
- Source columns: 21
- Target: `Churn` (`Yes` / `No`)
- Raw acquisition URL used by this repository: `https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv`

The raw CSV is downloaded at runtime into `data/raw/` and is intentionally excluded from Git.

## Claim boundaries

The dataset is a fictional/public sample, not proprietary production data. Metrics in this repository demonstrate modeling methodology on this sample only. The threshold-optimization cost values are explicit scenario assumptions for decision analysis; they are not IBM or telecom-company financial figures.

## Leakage policy

`customerID` is retained only to prove split disjointness and is excluded from model features. The target `Churn` is never used in feature engineering. All preprocessing is fitted on training data through scikit-learn pipelines.
