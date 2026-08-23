from pathlib import Path
import json

from src.train import main


def test_training_pipeline_end_to_end():
    main()
    root = Path(__file__).resolve().parents[1]
    metrics_path = root / "artifacts" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())

    selected = metrics["selection_policy"]["selected_model"]
    assert selected in {"logistic_regression", "random_forest"}
    assert metrics["selection_policy"]["metric"] == "validation_pr_auc"
    assert metrics["explainability"]["model"] == selected
    assert metrics["explainability"]["method"] in {"LinearExplainer", "TreeExplainer"}

    threshold = metrics["threshold_policy"]["chosen_threshold"]
    assert 0.10 <= threshold <= 0.90
    assert metrics["threshold_policy"]["selected_on"] == "validation only"
    assert metrics["threshold_policy"]["costs_are"] == "illustrative scenario assumptions"

    assert metrics["test_metrics_at_0_5"]["roc_auc"] > 0.70
    assert metrics["test_metrics_at_0_5"]["pr_auc"] > 0.45
    assert metrics["test_metrics"]["roc_auc"] > 0.70
    assert metrics["test_metrics"]["pr_auc"] > 0.45
    assert metrics["split_sizes"] == {"train": 4225, "validation": 1409, "test": 1409}

    assert (root / "artifacts" / "threshold_search.csv").exists()
    assert (root / "artifacts" / "subgroup_metrics.csv").exists()
    assert (root / "artifacts" / "shap_feature_importance.csv").exists()
    assert (root / "artifacts" / "model.joblib").exists()
