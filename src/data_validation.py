"""Validation helpers for the research-paper training dataset."""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence


TARGET_FIELDS = ("manual_score", "novelty_score", "rigor_score", "impact_score")
NUMERIC_FIELDS = (
    "author_count",
    "summary_len",
    "text_len",
    "years_since_published",
    *TARGET_FIELDS,
)


class DatasetValidationError(ValueError):
    """Raised when the dataset cannot safely be used for training or evaluation."""


@dataclass(frozen=True)
class DatasetValidationReport:
    """Summary of structural and label-quality checks."""

    record_count: int
    valid_embedding_count: int
    missing_target_counts: Dict[str, int]
    zero_target_counts: Dict[str, int]
    category_counts: Dict[str, int]

    @property
    def complete_label_count(self) -> int:
        return self.record_count - max(self.missing_target_counts.values(), default=0)

    @property
    def all_zero_label_count(self) -> int:
        return self.zero_target_counts.get("manual_score", 0)


def validate_records(
    records: Sequence[Dict[str, Any]],
    *,
    embedding_dimension: int = 384,
    require_embeddings: bool = True,
    require_targets: bool = False,
) -> DatasetValidationReport:
    """Validate records and return label-quality statistics.

    Structural errors raise ``DatasetValidationError``. Zero labels are retained
    because they may be valid, but are reported explicitly for review.
    """
    if not records:
        raise DatasetValidationError("Dataset is empty.")

    missing_targets = Counter()
    zero_targets = Counter()
    categories = Counter()
    valid_embeddings = 0
    errors: List[str] = []

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            errors.append(f"record {index}: expected an object")
            continue

        category = record.get("primary_category")
        if not isinstance(category, str) or not category.strip():
            errors.append(f"record {index}: primary_category must be a non-empty string")
        else:
            categories[category] += 1

        for field in NUMERIC_FIELDS:
            value = record.get(field)
            if value is None:
                if field in TARGET_FIELDS:
                    missing_targets[field] += 1
                elif require_targets or require_embeddings:
                    errors.append(f"record {index}: {field} is missing")
                if field in TARGET_FIELDS and require_targets:
                    errors.append(f"record {index}: {field} is missing")
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                errors.append(f"record {index}: {field} must be numeric")
            elif not math.isfinite(float(value)):
                errors.append(f"record {index}: {field} must be finite")
            elif field in TARGET_FIELDS and float(value) == 0:
                zero_targets[field] += 1

        embedding = record.get("text_embedding")
        if embedding is None:
            if require_embeddings:
                errors.append(f"record {index}: text_embedding is missing")
        elif not isinstance(embedding, list) or len(embedding) != embedding_dimension:
            errors.append(
                f"record {index}: text_embedding must contain "
                f"{embedding_dimension} values"
            )
        elif any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            for value in embedding
        ):
            errors.append(f"record {index}: text_embedding contains invalid values")
        else:
            valid_embeddings += 1

    if errors:
        preview = "; ".join(errors[:5])
        suffix = f" ({len(errors)} total errors)" if len(errors) > 5 else ""
        raise DatasetValidationError(f"Dataset validation failed: {preview}{suffix}")

    return DatasetValidationReport(
        record_count=len(records),
        valid_embedding_count=valid_embeddings,
        missing_target_counts=dict(missing_targets),
        zero_target_counts=dict(zero_targets),
        category_counts=dict(categories),
    )


def is_unlabeled_record(record: Dict[str, Any]) -> bool:
    """Return whether all score fields are zero placeholders."""
    if record.get("label_source") == "manual_annotation":
        return False
    return all(float(record.get(field, 0) or 0) == 0 for field in TARGET_FIELDS)
