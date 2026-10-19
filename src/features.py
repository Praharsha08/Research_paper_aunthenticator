"""
Module for feature engineering, text cleaning, and preprocessing research paper metadata.
"""

import json
import logging
from typing import List, Dict, Any, Tuple
import pandas as pd
import numpy as np

from src.data_validation import is_unlabeled_record

logger = logging.getLogger(__name__)


def load_categories(dataset_path: str = "data/embeddingdataset.json") -> List[str]:
    """
    Dynamically loads all unique primary categories present in the dataset.
    Falls back to a default set of ML/CS categories if dataset is unavailable.

    Args:
        dataset_path: Path to the JSON dataset file.

    Returns:
        Sorted list of unique category strings.
    """
    default_categories = ['cs.NE', 'cs.LG', 'cs.CV', 'stat.ML', 'cs.CL', 'unknown']
    try:
        with open(dataset_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        categories = set()
        for record in data:
            cat = record.get("primary_category")
            if cat:
                categories.add(cat)
                
        categories_list = sorted(list(categories))
        if "unknown" not in categories_list:
            categories_list.append("unknown")
        return categories_list
    except Exception as e:
        logger.warning(f"Could not load categories dynamically: {e}. Using defaults.")
        return default_categories


def clean_abstract(text: str) -> str:
    """
    Extracts the abstract section from a research paper's raw text.
    Locates the text segment between 'Abstract' and 'Introduction' headers.
    Falls back to the first 1500 characters if markers are missing.

    Args:
        text: Raw text extracted from the PDF.

    Returns:
        Cleaned abstract text.
    """
    lower_text = text.lower()
    start_idx = lower_text.find("abstract")
    
    if start_idx != -1:
        start_idx += 8  # Skip word 'abstract'
        
        # Look for standard section dividers
        end_idx = lower_text.find("introduction", start_idx)
        if end_idx == -1:
            end_idx = lower_text.find("1. ", start_idx)
        if end_idx == -1:
            end_idx = start_idx + 1500  # Fallback offset
            
        abstract_text = text[start_idx:end_idx].strip()
        # Clean leading noise
        if abstract_text.startswith(":") or abstract_text.startswith("\n"):
            abstract_text = abstract_text[1:].strip()
        return abstract_text
        
    return text[:1500].strip()


def build_dataset(
    data: List[Dict[str, Any]],
    categories: List[str] | None = None,
) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Constructs the feature matrix and target arrays from raw records for model training.

    Args:
        data: List of raw document records containing manual scores, embeddings, and metadata.

    Returns:
        X (DataFrame): Preprocessed features (scalars, categories, and embeddings).
        y_overall (ndarray): Target overall scores.
        y_novelty (ndarray): Target novelty sub-scores.
        y_rigor (ndarray): Target rigor sub-scores.
        y_impact (ndarray): Target impact sub-scores.
    """
    rows = []
    y_overall = []
    y_novelty = []
    y_rigor = []
    y_impact = []
    
    # Pre-load categories to align pandas category codes
    all_categories = categories if categories is not None else load_categories()

    for record in data:
        # Require embeddings and manual score for training
        if (
            record.get("text_embedding") is None
            or record.get("manual_score") is None
            or is_unlabeled_record(record)
        ):
            continue
            
        # Scalars
        row = {
            "author_count": float(record.get("author_count", 0) or 0),
            "summary_len": float(record.get("summary_len", 0) or 0),
            "text_len": float(record.get("text_len", 0) or 0),
            "years_since_published": float(record.get("years_since_published", 0) or 0),
            "primary_category": record.get("primary_category", "unknown")
        }
        
        # Dense 384-dimensional sentence embeddings
        embedding = record["text_embedding"]
        for i, val in enumerate(embedding):
            row[f"emb_{i}"] = float(val)
            
        rows.append(row)
        
        y_overall.append(float(record["manual_score"]))
        y_novelty.append(float(record.get("novelty_score", 0)))
        y_rigor.append(float(record.get("rigor_score", 0)))
        y_impact.append(float(record.get("impact_score", 0)))
        
    df = pd.DataFrame(rows)
    # Ensure pandas category codes alignment
    df["primary_category"] = pd.Categorical(df["primary_category"], categories=all_categories)
    
    return df, np.array(y_overall), np.array(y_novelty), np.array(y_rigor), np.array(y_impact)


def prepare_features(metadata: Dict[str, Any], embedding: List[float], categories: List[str]) -> pd.DataFrame:
    """
    Preprocesses a single paper instance into the exact DataFrame schema required by the model.

    Args:
        metadata: Dictionary containing metadata features (author count, lengths, years).
        embedding: List of 384 floats representing the text embedding.
        categories: List of categories to align categorical codes.

    Returns:
        DataFrame containing one formatted row of features.
    """
    row = {
        "author_count": float(metadata.get("author_count", 0) or 0),
        "summary_len": float(metadata.get("summary_len", 0) or 0),
        "text_len": float(metadata.get("text_len", 0) or 0),
        "years_since_published": float(metadata.get("years_since_published", 0) or 0),
        "primary_category": metadata.get("primary_category", "unknown")
    }
    
    # Inject embedding features
    for i, val in enumerate(embedding):
        row[f"emb_{i}"] = float(val)
        
    df = pd.DataFrame([row])
    # Convert column to category explicitly after DataFrame creation
    df["primary_category"] = pd.Categorical(df["primary_category"], categories=categories)
    
    # Establish strict column ordering matching the trained dataset
    emb_cols = [f"emb_{i}" for i in range(len(embedding))]
    cols_order = ["author_count", "summary_len", "text_len", "years_since_published", "primary_category"] + emb_cols
    return df[cols_order]
