"""Validate annotations and build a model-compatible enriched dataset."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List

ANNOTATIONS_PATH = Path("data/annotations.json")
SOURCE_DATASET_PATH = Path("data/embeddingdataset.json")
OUTPUT_DATASET_PATH = Path("data/annotated_embeddingdataset.json")
SCORE_FIELDS = ("overall", "novelty", "rigor", "impact")
TARGET_FIELDS = ("manual_score", "novelty_score", "rigor_score", "impact_score")
TARGET_MAXIMUMS = (100.0, 35.0, 35.0, 35.0)


class AnnotationImportError(ValueError):
    """Raised when annotations cannot be safely imported."""


def load_json(path: Path) -> Any:
    if not path.exists():
        raise AnnotationImportError(f"Required file not found: {path}")
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError as error:
        raise AnnotationImportError(f"Invalid JSON in {path}: {error}") from error


def validate_annotations(raw_annotations: Any) -> Iterable[Dict[str, Any]]:
    if not isinstance(raw_annotations, dict):
        raise AnnotationImportError("Annotations must be a JSON object keyed by ArXiv ID.")

    for arxiv_id, annotation in raw_annotations.items():
        if not isinstance(arxiv_id, str) or not arxiv_id.strip():
            raise AnnotationImportError("Annotation keys must be non-empty ArXiv IDs.")
        if not isinstance(annotation, dict):
            raise AnnotationImportError(f"Annotation for {arxiv_id} must be an object.")

        for field in SCORE_FIELDS:
            value = annotation.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise AnnotationImportError(
                    f"Annotation {arxiv_id}.{field} must be a number from 1 to 5."
                )
            if not 1 <= float(value) <= 5:
                raise AnnotationImportError(
                    f"Annotation {arxiv_id}.{field} must be between 1 and 5."
                )

        yield {"arxiv_id": arxiv_id, **annotation}


def convert_score(score: float, maximum: float) -> float:
    """Convert an inclusive 1–5 rating to the target range."""
    return round((float(score) - 1.0) * maximum / 4.0, 4)


def build_dataset(
    source_records: List[Dict[str, Any]],
    annotations: Iterable[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], int]:
    records_by_id = {record.get("arxiv_id"): record for record in source_records}
    imported = 0
    output_records = [dict(record) for record in source_records]

    for annotation in annotations:
        arxiv_id = annotation["arxiv_id"]
        record = records_by_id.get(arxiv_id)
        if record is None:
            raise AnnotationImportError(
                f"Annotation references unknown ArXiv ID: {arxiv_id}"
            )
        if record.get("text_embedding") is None:
            raise AnnotationImportError(
                f"Annotated record has no embedding and cannot be trained: {arxiv_id}"
            )
        if any(record.get(field) not in (None, 0) for field in TARGET_FIELDS):
            raise AnnotationImportError(
                f"Record already has a non-zero label and cannot be overwritten: {arxiv_id}"
            )

        enriched = dict(record)
        for score_field, target_field, maximum in zip(
            SCORE_FIELDS, TARGET_FIELDS, TARGET_MAXIMUMS
        ):
            enriched[target_field] = convert_score(annotation[score_field], maximum)
        enriched["label_source"] = "manual_annotation"
        enriched["annotation"] = {
            "scale": "1-5",
            "annotated_at": annotation.get(
                "annotated_at", datetime.now(timezone.utc).isoformat()
            ),
            "notes": annotation.get("notes", ""),
        }
        output_records[output_records.index(record)] = enriched
        imported += 1

    return output_records, imported


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import 1–5 Streamlit annotations into a separate training dataset."
    )
    parser.add_argument("--annotations", type=Path, default=ANNOTATIONS_PATH)
    parser.add_argument("--source", type=Path, default=SOURCE_DATASET_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_DATASET_PATH)
    args = parser.parse_args()

    source_records = load_json(args.source)
    if not isinstance(source_records, list) or not all(
        isinstance(record, dict) for record in source_records
    ):
        raise AnnotationImportError("Source dataset must be a JSON list of objects.")

    annotations = list(validate_annotations(load_json(args.annotations)))
    output_records, imported = build_dataset(source_records, annotations)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        json.dump(output_records, file, indent=2)

    print(f"Imported annotations: {imported}")
    print(f"Output records: {len(output_records)}")
    print(f"Wrote: {args.output}")


if __name__ == "__main__":
    try:
        main()
    except AnnotationImportError as error:
        raise SystemExit(f"Annotation import failed: {error}")
