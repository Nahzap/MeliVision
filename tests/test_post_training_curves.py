"""Regression: plot_training_curves must accept new training_metrics.csv schema."""

import tempfile
from pathlib import Path

import pandas as pd

from scripts.post_training import plot_training_curves


def _write_new_schema_csv(path: Path) -> None:
    rows = [
        [1, 1.0, 1.2, "mAP@R", 0.47, 0.83, 0.95, 0.97, 0.47, 0.60, 0.83, 0.83, 0.82, 0.84, 0.84, 0.83, 1e-4, True, 0.47, 100.0],
        [2, 0.8, 1.0, "mAP@R", 0.66, 0.90, 0.97, 0.98, 0.66, 0.76, 0.90, 0.90, 0.90, 0.90, 0.90, 0.90, 5e-5, True, 0.66, 95.0],
    ]
    cols = [
        "epoch", "train_loss", "val_loss", "selection_metric", "selection_value",
        "recall_at_1", "recall_at_5", "recall_at_10", "mAP@R", "NMI", "F1_macro",
        "accuracy", "balanced_accuracy", "precision_macro", "precision_weighted",
        "F1_weighted", "learning_rate", "is_best", "best_selection_value", "epoch_time_sec",
    ]
    pd.DataFrame(rows, columns=cols).to_csv(path, index=False)


def test_plot_training_curves_new_csv_schema():
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = Path(tmp) / "training_metrics.csv"
        out_dir = Path(tmp) / "plots"
        _write_new_schema_csv(csv_path)
        plot_training_curves(str(csv_path), str(out_dir))
        assert (out_dir / "loss_curves.png").exists()
        assert (out_dir / "recall_curves.png").exists()
        assert (out_dir / "lr_curve.png").exists()
