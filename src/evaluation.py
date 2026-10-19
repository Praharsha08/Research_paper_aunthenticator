"""Leakage-safe, out-of-fold evaluation for trained feature pipelines."""

from __future__ import annotations

from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold

from src.features import build_dataset
from src.data_validation import is_unlabeled_record


TARGET_NAMES = {
    "overall": "manual_score",
    "novelty": "novelty_score",
    "rigor": "rigor_score",
    "impact": "impact_score",
}


def create_regressor() -> lgb.LGBMRegressor:
    """Return the single model configuration used by training and evaluation."""
    return lgb.LGBMRegressor(
        n_estimators=80,
        learning_rate=0.04,
        max_depth=3,
        num_leaves=7,
        min_child_samples=4,
        random_state=42,
        verbosity=-1,
    )


def out_of_fold_evaluate(
    data: list,
    *,
    n_splits: int = 5,
    random_state: int = 42,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, np.ndarray], np.ndarray]:
    """Generate out-of-fold predictions and metrics for every target.

    Every prediction is produced by a model that did not train on that record.
    """
    labelled_data = [record for record in data if not is_unlabeled_record(record)]
    if len(labelled_data) != len(data):
        # Keep evaluation aligned with the records that can provide meaningful targets.
        data = labelled_data
    X, y_overall, y_novelty, y_rigor, y_impact = build_dataset(data)
    targets = {
        "overall": y_overall,
        "novelty": y_novelty,
        "rigor": y_rigor,
        "impact": y_impact,
    }
    if len(X) < n_splits:
        raise ValueError(f"Need at least {n_splits} labelled records for evaluation.")

    predictions = {name: np.full(len(X), np.nan) for name in targets}
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)

    for train_indices, validation_indices in splitter.split(X):
        for name, target in targets.items():
            model = create_regressor()
            model.fit(X.iloc[train_indices], target[train_indices])
            predictions[name][validation_indices] = model.predict(X.iloc[validation_indices])

    metrics: Dict[str, Dict[str, float]] = {}
    for name, target in targets.items():
        errors = np.abs(target - predictions[name])
        metrics[name] = {
            "mae": float(mean_absolute_error(target, predictions[name])),
            "rmse": float(np.sqrt(mean_squared_error(target, predictions[name]))),
            "r2": float(r2_score(target, predictions[name])),
            "within_5_percent": float(np.mean(errors <= 5) * 100),
            "within_10_percent": float(np.mean(errors <= 10) * 100),
        }

    return metrics, predictions, y_overall
