"""
Script to train overall and sub-score LightGBM regressors on the paper embedding dataset.
"""

import json
import logging
from pathlib import Path
import lightgbm as lgb
import joblib
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error
import numpy as np
import pandas as pd

from src.features import build_dataset

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = Path("data/embeddingdataset.json")
MODELS_DIR = Path("models")


def load_data(path: Path) -> list:
    """
    Loads JSON dataset from the specified path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run embed_all.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def train_and_evaluate(X: pd.DataFrame, y: np.ndarray, target_name: str) -> lgb.LGBMRegressor:
    """
    Trains a LightGBM Regressor using 5-fold cross-validation and returns the trained model.
    """
    # Conservative hyperparameters to prevent overfitting on small n=181 dataset
    model = lgb.LGBMRegressor(
        n_estimators=80,
        learning_rate=0.04,
        max_depth=3,
        num_leaves=7,
        min_child_samples=4,
        random_state=42,
        verbosity=-1
    )
    
    # 5-Fold Cross Validation
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    maes = []
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Clone model configuration for this fold
        fold_model = lgb.LGBMRegressor(**model.get_params())
        fold_model.fit(X_train, y_train)
        
        preds = fold_model.predict(X_val)
        mae = mean_absolute_error(y_val, preds)
        maes.append(mae)
        
    mean_mae = np.mean(maes)
    logger.info(f"[{target_name}] Mean CV MAE (5-Fold): {mean_mae:.3f} | MAE per fold: {[f'{x:.2f}' for x in maes]}")
    
    # Fit model on all data
    model.fit(X, y)
    return model


def main() -> None:
    logger.info(f"Loading embedding dataset from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    
    logger.info("Preprocessing records into features...")
    X, y_overall, y_novelty, y_rigor, y_impact = build_dataset(data)
    logger.info(f"Feature matrix shape: {X.shape}, samples: {len(X)}")
    
    # Train each model
    logger.info("\nTraining Overall Score Model...")
    overall_model = train_and_evaluate(X, y_overall, "Overall Quality Score")
    
    logger.info("\nTraining Novelty Sub-Score Model...")
    novelty_model = train_and_evaluate(X, y_novelty, "Novelty Sub-Score")
    
    logger.info("\nTraining Rigor Sub-Score Model...")
    rigor_model = train_and_evaluate(X, y_rigor, "Rigor Sub-Score")
    
    logger.info("\nTraining Impact Sub-Score Model...")
    impact_model = train_and_evaluate(X, y_impact, "Impact Sub-Score")
    
    # Save the models
    MODELS_DIR.mkdir(exist_ok=True)
    logger.info(f"\nSaving models to {MODELS_DIR}...")
    joblib.dump(overall_model, MODELS_DIR / "lightgbmv1.joblib")
    joblib.dump(novelty_model, MODELS_DIR / "novelty_model.joblib")
    joblib.dump(rigor_model, MODELS_DIR / "rigor_model.joblib")
    joblib.dump(impact_model, MODELS_DIR / "impact_model.joblib")
    
    logger.info("All models trained and saved successfully!")


if __name__ == "__main__":
    main()
