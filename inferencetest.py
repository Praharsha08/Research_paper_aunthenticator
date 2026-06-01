"""
CLI script to load trained LightGBM models and evaluate performance on the full embedded dataset.
Outputs global metrics, category-wise breakdown, and runs a terminal SHAP visualization.
"""

import json
import logging
import time
from pathlib import Path
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np

from src.features import prepare_features, load_categories
from src.predictor import ResearchPaperPredictor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Paths
DATASET_PATH = Path("data/embeddingdataset.json")


def load_data(path: Path) -> list:
    """
    Loads JSON dataset from path.
    """
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}. Run train.py first.")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    # Initialize the predictor
    predictor = ResearchPaperPredictor(models_dir="models")
    if not predictor.is_ready():
        logger.error("Predictor models not found in models/ directory. Run train.py first.")
        return

    logger.info(f"Loading dataset from {DATASET_PATH}...")
    data = load_data(DATASET_PATH)
    
    # Load dynamic categories
    categories = load_categories(str(DATASET_PATH))
    
    # Process all valid records
    y_true = []
    y_pred = []
    processed_records = []
    
    t0 = time.perf_counter()
    for record in data:
        if record.get("manual_score") is not None and record.get("text_embedding") is not None:
            # Prepare metadata
            metadata = {
                "author_count": record.get("author_count", 0),
                "summary_len": record.get("summary_len", 0),
                "text_len": record.get("text_len", 0),
                "years_since_published": record.get("years_since_published", 0),
                "primary_category": record.get("primary_category", "unknown")
            }
            embedding = record["text_embedding"]
            
            # Format row
            X = prepare_features(metadata, embedding, categories)
            
            # Predict overall score
            pred_scores = predictor.predict_scores(X)
            prediction = pred_scores["overall"]
            
            y_true.append(float(record["manual_score"]))
            y_pred.append(prediction)
            
            processed_records.append({
                "arxiv_id": record.get("arxiv_id", "Unknown"),
                "title": record.get("title", "Untitled"),
                "category": record.get("primary_category", "unknown"),
                "actual": float(record["manual_score"]),
                "predicted": prediction,
                "diff": abs(float(record["manual_score"]) - prediction),
                "X": X
            })
    t1 = time.perf_counter()
    
    n_samples = len(y_true)
    if n_samples == 0:
        logger.error("No valid papers with embeddings and manual scores found.")
        return
        
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    # Calculate global metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Within thresholds
    diffs = np.abs(y_true - y_pred)
    within_5 = np.sum(diffs <= 5) / n_samples * 100
    within_10 = np.sum(diffs <= 10) / n_samples * 100
    
    print("\n" + "=" * 50)
    print(f"         MODEL EVALUATION METRICS ({n_samples} papers)")
    print("=" * 50)
    print(f"Mean Absolute Error (MAE):      {mae:.3f} points")
    print(f"Root Mean Squared Error (RMSE): {rmse:.3f} points")
    print(f"R² Score (Variance Explained):   {r2:.3f}")
    print(f"Predictions within ±5 points:   {within_5:.1f}%")
    print(f"Predictions within ±10 points:  {within_10:.1f}%")
    print(f"Total evaluation time:          {(t1 - t0) * 1000:.2f} ms")
    print("=" * 50)
    
    # Category-wise performance breakdown
    print("\n--- Performance Breakdown by Category ---")
    cat_groups = {}
    for r in processed_records:
        cat_groups.setdefault(r["category"], []).append(r["diff"])
        
    print(f"{'Category':<15} | {'Count':<6} | {'Mean Absolute Error (MAE)':<25}")
    print("-" * 55)
    for cat, diff_list in sorted(cat_groups.items(), key=lambda x: len(x[1]), reverse=True):
        cat_mae = np.mean(diff_list)
        print(f"{cat:<15} | {len(diff_list):<6} | {cat_mae:.3f}")
    print("-" * 55)
    
    # Best & Worst Predictions
    sorted_records = sorted(processed_records, key=lambda x: x["diff"])
    
    print("\n--- Top 3 Best Predictions (Closest Matches) ---")
    for i, r in enumerate(sorted_records[:3]):
        print(f"{i + 1}. ArXiv ID: {r['arxiv_id']} | Actual: {r['actual']} | Predicted: {r['predicted']:.2f} | Diff: {r['diff']:.2f}")
        print(f"   Title: {r['title']}")
        
    print("\n--- Top 3 Worst Predictions (Largest Errors) ---")
    for i, r in enumerate(sorted_records[-3:][::-1]):
        print(f"{i + 1}. ArXiv ID: {r['arxiv_id']} | Actual: {r['actual']} | Predicted: {r['predicted']:.2f} | Diff: {r['diff']:.2f}")
        print(f"   Title: {r['title']}")
        
    # Run SHAP on the first paper as a demo
    if len(processed_records) > 0:
        sample = processed_records[0]
        print(f"\nGenerating terminal SHAP demonstration for: '{sample['title']}'")
        
        # We can extract TreeSHAP values locally in CLI
        try:
            import shap
            explainer = shap.TreeExplainer(predictor.models["overall"])
            shap_values = explainer.shap_values(sample["X"])
            
            # Aggregate embedding SHAP values
            sv_scalars = {
                "Author Count": shap_values[0, 0],
                "Summary Len": shap_values[0, 1],
                "Text Length": shap_values[0, 2],
                "Years Published": shap_values[0, 3],
                "Primary Category": shap_values[0, 4]
            }
            sv_embeddings = np.sum(shap_values[0, 5:])
            sv_scalars["Text Semantics (Emb)"] = sv_embeddings
            
            print("\n=== SHAP Feature Contributions (Terminal Explainer) ===")
            print(f"Base Value (Dataset Mean Expected): {explainer.expected_value:.2f}")
            print(f"Model Prediction:                 {sample['predicted']:.2f}")
            print(f"Actual Paper Score:               {sample['actual']:.2f}")
            print("-" * 55)
            
            # Sort by contribution magnitude
            sorted_features = sorted(sv_scalars.items(), key=lambda x: abs(x[1]), reverse=True)
            for name, val in sorted_features:
                bar_len = int(abs(val) * 1.5)
                bar_char = "+" if val >= 0 else "-"
                bar_str = bar_char * bar_len
                print(f"{name:<25} | {val:>6.2f} | {bar_str}")
            print("=" * 55)
        except Exception as e:
            print(f"Could not calculate SHAP values: {e}")


if __name__ == "__main__":
    main()
