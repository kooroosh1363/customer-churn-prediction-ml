from pathlib import Path
import json

from src.train import main


def test_training_pipeline_end_to_end():
    main()
    root = Path(__file__).resolve().parents[1]
    metrics_path = root / "artifacts" / "metrics.json"
    assert metrics_path.exists()
    metrics = json.loads(metrics_path.read_text())
    assert metrics["selected_model"] in {"logistic_regression", "random_forest"}
    assert 0.0 < metrics["threshold_policy"]["chosen_threshold"] < 1.0
    assert metrics["test_metrics"]["roc_auc"] > 0.70
    assert metrics["test_metrics"]["pr_auc"] > 0.45
    assert metrics["split_sizes"] == {"train": 4225, "validation": 1409, "test": 1409}
