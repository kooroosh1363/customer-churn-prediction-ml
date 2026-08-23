from __future__ import annotations

from pathlib import Path
import requests
import pandas as pd

URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Telco-Customer-Churn.csv"
EXPECTED_COLUMNS = [
    "customerID","gender","SeniorCitizen","Partner","Dependents","tenure",
    "PhoneService","MultipleLines","InternetService","OnlineSecurity",
    "OnlineBackup","DeviceProtection","TechSupport","StreamingTV",
    "StreamingMovies","Contract","PaperlessBilling","PaymentMethod",
    "MonthlyCharges","TotalCharges","Churn"
]


def load_data() -> pd.DataFrame:
    RAW.parent.mkdir(parents=True, exist_ok=True)
    if not RAW.exists():
        response = requests.get(URL, timeout=60)
        response.raise_for_status()
        RAW.write_bytes(response.content)
    df = pd.read_csv(RAW)
    if list(df.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")
    if len(df) != 7043:
        raise ValueError(f"Unexpected row count: {len(df)}")
    if df["customerID"].duplicated().any():
        raise ValueError("customerID must be unique")
    if set(df["Churn"].dropna().unique()) != {"Yes", "No"}:
        raise ValueError("Unexpected target labels")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["TotalCharges"] = pd.to_numeric(out["TotalCharges"], errors="coerce")
    out["avg_monthly_value"] = out["TotalCharges"] / out["tenure"].replace(0, pd.NA)
    service_cols = [
        "PhoneService","OnlineSecurity","OnlineBackup","DeviceProtection",
        "TechSupport","StreamingTV","StreamingMovies"
    ]
    out["service_count"] = sum((out[c] == "Yes").astype(int) for c in service_cols)
    out["is_month_to_month"] = (out["Contract"] == "Month-to-month").astype(int)
    out["uses_auto_payment"] = out["PaymentMethod"].str.contains("automatic", case=False, na=False).astype(int)
    return out
