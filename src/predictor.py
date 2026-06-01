"""
Module for loading model binaries, conducting regression inference, and computing explainability maps (SHAP).
"""

from typing import Dict, Any, Optional
import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional SHAP import to support graceful fallback
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False


class ResearchPaperPredictor:
    """
    Wrapper interface to manage and evaluate LightGBM regression models for scoring papers.
    """

    def __init__(self, models_dir: str = "models") -> None:
        """
        Initializes the predictor and attempts to load all four LightGBM model joblibs.

        Args:
            models_dir: Folder containing model binaries.
        """
        self.models_dir = models_dir
        self.models: Dict[str, Any] = {}
        self.load_models()

    def load_models(self) -> None:
        """
        Loads models from disk. Populates self.models dictionary.
        """
        model_names = {
            "overall": "lightgbmv1.joblib",
            "novelty": "novelty_model.joblib",
            "rigor": "rigor_model.joblib",
            "impact": "impact_model.joblib"
        }
        
        for key, filename in model_names.items():
            path = os.path.join(self.models_dir, filename)
            if os.path.exists(path):
                self.models[key] = joblib.load(path)
            else:
                # Try fallback parent directory in case of execution context changes
                parent_path = os.path.join(filename)
                if os.path.exists(parent_path):
                    self.models[key] = joblib.load(parent_path)

    def is_ready(self) -> bool:
        """
        Checks if all four regression models are loaded and available.

        Returns:
            True if all models are loaded, False otherwise.
        """
        required_keys = ["overall", "novelty", "rigor", "impact"]
        return all(key in self.models for key in required_keys)

    def predict_scores(self, X: pd.DataFrame) -> Dict[str, float]:
        """
        Predicts overall score, novelty, rigor, and impact for a structured feature row.

        Args:
            X: Input DataFrame containing formatted paper features.

        Returns:
            Dictionary containing predicted float values for all 4 targets.
        """
        if not self.is_ready():
            raise ValueError("Predictor is not fully loaded. Ensure models are trained and present in models/.")
            
        pred_overall = max(0.0, min(100.0, float(self.models["overall"].predict(X)[0])))
        pred_novelty = max(0.0, min(35.0, float(self.models["novelty"].predict(X)[0])))
        pred_rigor = max(0.0, min(35.0, float(self.models["rigor"].predict(X)[0])))
        pred_impact = max(0.0, min(35.0, float(self.models["impact"].predict(X)[0])))
        
        return {
            "overall": pred_overall,
            "novelty": pred_novelty,
            "rigor": pred_rigor,
            "impact": pred_impact
        }

    def generate_shap_plot(self, X: pd.DataFrame) -> plt.Figure:
        """
        Generates a horizontal bar chart of local SHAP value contributions.
        Aggregates the 384 embedding vector SHAP values into a single semantic score.
        Falls back to model feature importances if SHAP is not installed.

        Args:
            X: Preprocessed DataFrame representing the target paper features.

        Returns:
            Matplotlib Figure object.
        """
        fig, ax = plt.subplots(figsize=(6, 3))
        
        if SHAP_AVAILABLE and "overall" in self.models:
            try:
                explainer = shap.TreeExplainer(self.models["overall"])
                shap_values = explainer.shap_values(X)
                
                # Extract and aggregate SHAP contributions
                sv_scalars = {
                    "Author Count": shap_values[0, 0],
                    "Abstract Length": shap_values[0, 1],
                    "Text Length": shap_values[0, 2],
                    "Years Published": shap_values[0, 3],
                    "ArXiv Category": shap_values[0, 4]
                }
                sv_embeddings = np.sum(shap_values[0, 5:])
                sv_scalars["Text Semantics (Embeddings)"] = sv_embeddings
                
                # Sort features by absolute contribution
                sorted_features = sorted(sv_scalars.items(), key=lambda x: abs(x[1]), reverse=True)
                names = [x[0] for x in sorted_features]
                values = [x[1] for x in sorted_features]
                
                colors = ['#38bdf8' if v >= 0 else '#f87171' for v in values]
                ax.barh(names, values, color=colors, edgecolor='#334155', height=0.6)
                ax.set_title("Feature Contribution to Score (SHAP)", color='#e2e8f0', fontsize=11, fontweight='bold')
                
            except Exception:
                self._plot_fallback_importances(ax)
        else:
            self._plot_fallback_importances(ax)

        # Apply general premium visual theme
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#475569')
        ax.spines['bottom'].set_color('#475569')
        ax.tick_params(colors='#e2e8f0', labelsize=9)
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        plt.tight_layout()
        return fig

    def _plot_fallback_importances(self, ax: plt.Axes) -> None:
        """
        Helper method to plot basic feature importances if SHAP fails.
        """
        if "overall" not in self.models:
            ax.text(0.5, 0.5, "Model not loaded.", color='#e2e8f0', ha='center', va='center')
            return
            
        importances = self.models["overall"].feature_importances_
        scalar_imp = {
            "Author Count": importances[0],
            "Abstract Length": importances[1],
            "Text Length": importances[2],
            "Years Published": importances[3],
            "ArXiv Category": importances[4],
            "Text Semantics": np.sum(importances[5:])
        }
        sorted_imp = sorted(scalar_imp.items(), key=lambda x: x[1], reverse=True)
        names = [x[0] for x in sorted_imp]
        values = [x[1] for x in sorted_imp]
        
        ax.barh(names, values, color='#818cf8', edgecolor='#334155', height=0.6)
        ax.set_title("Model Feature Importance (SHAP Unavailable)", color='#e2e8f0', fontsize=11, fontweight='bold')
