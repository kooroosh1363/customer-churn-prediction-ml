from src.data import load_data, engineer_features


def test_source_contract_and_target():
    df = load_data()
    assert len(df) == 7043
    assert df["customerID"].is_unique
    assert set(df["Churn"].unique()) == {"Yes", "No"}


def test_feature_engineering_has_no_target_leakage():
    df = load_data()
    out = engineer_features(df.drop(columns=["Churn"]))
    assert "Churn" not in out.columns
    assert {"avg_monthly_value", "service_count", "is_month_to_month", "uses_auto_payment"}.issubset(out.columns)
    assert out["service_count"].between(0, 7).all()
