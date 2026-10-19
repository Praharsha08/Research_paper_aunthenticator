"""
CLI script to load trained LightGBM models and evaluate performance on the full embedded dataset.
Outputs global metrics, category-wise breakdown, and runs a terminal SHAP visualization.
"""

import json
import logging
import time
from pathlib import Path
from src.data_validation import DatasetValidationError, is_unlabeled_record, validate_records
from src.evaluation import out_of_fold_evaluate

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DEFAULT_DATASET_PATH = Path("data/embeddingdataset.json")
ANNOTATED_DATASET_PATH = Path("data/annotated_embeddingdataset.json")


def load_data(path: Path) -> list:
    """
    Loads JSON dataset from path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run train.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    dataset_path = (
        ANNOTATED_DATASET_PATH
        if ANNOTATED_DATASET_PATH.exists()
        else DEFAULT_DATASET_PATH
    )
    logger.info("Loading dataset from %s...", dataset_path)
    data = load_data(dataset_path)
    try:
        report = validate_records(data, require_targets=True)
    except DatasetValidationError as exc:
        logger.error(str(exc))
        raise
    logger.info("Dataset validation summary: %s", report)
    unlabeled_count = sum(is_unlabeled_record(record) for record in data)
    if unlabeled_count:
        logger.warning(
            "Excluding %d all-zero records as unlabeled placeholders.",
            unlabeled_count,
        )

    t0 = time.perf_counter()
    metrics, predictions, overall_targets = out_of_fold_evaluate(data)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    
    print("\n" + "=" * 50)
    print(f"     OUT-OF-FOLD MODEL EVALUATION ({len(overall_targets)} papers)")
    print("=" * 50)
    print("Each prediction was generated without training on that record.")
    print("-" * 50)
    for name, values in metrics.items():
        print(
            f"{name.title():<8} | MAE: {values['mae']:.3f} | "
            f"RMSE: {values['rmse']:.3f} | R²: {values['r2']:.3f} | "
            f"±5: {values['within_5_percent']:.1f}% | "
            f"±10: {values['within_10_percent']:.1f}%"
        )
    print(f"Total evaluation time:          {elapsed_ms:.2f} ms")
    print("=" * 50)


if __name__ == "__main__":
    main()
