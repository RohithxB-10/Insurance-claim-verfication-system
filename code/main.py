"""
Insurance Claim Verification System — Main Entry Point

Orchestrates the end-to-end pipeline:
  1. Load datasets (claims, user history, evidence requirements)
  2. For each claim: run claim analysis, image analysis, and final review
  3. Build output rows matching the output.csv schema
  4. Save results to dataset/output.csv

Usage:
    python code/main.py

Requirements:
    - Python 3.11+
    - OPENAI_API_KEY / ANTHROPIC_API_KEY env vars (if AI integration is added later)
    - All dataset files present under dataset/

No AI model calls are made in this version; all analysis is rule-based.
"""

from __future__ import annotations

import csv
import logging
import sys
import time
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Sys-path bootstrapping
# Ensure the code/ directory is on the path so `engines.*` and `utils.*`
# imports resolve correctly regardless of the working directory used to invoke
# this script.
# ---------------------------------------------------------------------------
_CODE_DIR = Path(__file__).resolve().parents[0]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import pandas as pd

from engines.claim_engine import analyze_claim
from engines.evidence_engine import analyze_images
from engines.review_engine import review_claim
from utils.csv_loader import (
    create_dataset_paths,
    load_claims,
    load_evidence_requirements,
    load_user_history,
)

# ---------------------------------------------------------------------------
# Logging configuration
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("claim_verification.main")

# ---------------------------------------------------------------------------
# Output schema (column order must match output.csv exactly)
# ---------------------------------------------------------------------------
OUTPUT_COLUMNS: list[str] = [
    "user_id",
    "image_paths",
    "user_claim",
    "claim_object",
    "evidence_standard_met",
    "evidence_standard_met_reason",
    "risk_flags",
    "issue_type",
    "object_part",
    "claim_status",
    "claim_status_justification",
    "supporting_image_ids",
    "valid_image",
    "severity",
]


# ---------------------------------------------------------------------------
# Image-path helper
# ---------------------------------------------------------------------------

def _rebase_image_paths(image_paths: str, dataset_dir: Path) -> str:
    """
    Rebase semicolon-separated relative image paths onto the dataset directory.

    claims.csv stores paths relative to dataset/ (e.g. "images/test/case_001/img_1.jpg").
    evidence_engine resolves paths with Path.resolve(), which anchors to CWD (repo root),
    producing the wrong absolute path.  Prepending dataset_dir corrects this before the
    paths reach the engine.

    Args:
        image_paths: Semicolon-separated path string from claims.csv.
        dataset_dir: Absolute Path to the dataset/ directory.

    Returns:
        Semicolon-separated string of absolute paths rooted at dataset_dir.
        Already-absolute paths are passed through unchanged.
    """
    rebased: list[str] = []
    for p in image_paths.split(";"):
        p = p.strip()
        if not p:
            continue
        path_obj = Path(p)
        if path_obj.is_absolute():
            rebased.append(str(path_obj))
        else:
            rebased.append(str(dataset_dir / path_obj))
    return ";".join(rebased)


# ---------------------------------------------------------------------------
# Row-level processor
# ---------------------------------------------------------------------------

def process_claim_row(
    row: pd.Series,
    user_history_map: dict[str, dict[str, Any]],
    evidence_requirements: dict[str, dict[str, Any]],
    dataset_dir: Path,
) -> dict[str, Any]:
    """
    Process a single claim row through all three analysis engines.

    Args:
        row: A pandas Series representing one row from claims.csv.
             Expected columns: user_id, image_paths, user_claim, claim_object.
        user_history_map: Dict keyed by user_id, from load_user_history().
        evidence_requirements: Dict keyed by requirement_id, from
                               load_evidence_requirements(). Passed through to
                               review_engine; the engine defaults to
                               min_images=1 for unknown structures.

    Returns:
        Dict whose keys match OUTPUT_COLUMNS exactly.

    Notes:
        - All exceptions within a row are caught; on failure the row is emitted
          with safe fallback values so the overall pipeline never halts.
        - image_paths are rebased onto dataset_dir before being passed to the
          evidence engine so that Path.resolve() produces correct absolute paths.
    """
    # ---- Extract fields from the CSV row ----------------------------------
    user_id: str = str(row.get("user_id", "")).strip()
    image_paths: str = str(row.get("image_paths", "")).strip()
    user_claim: str = str(row.get("user_claim", "")).strip()
    claim_object: str = str(row.get("claim_object", "")).strip().lower()

    logger.debug("Processing claim for user_id=%s", user_id)

    # ---- Resolve user history (may be absent for new users) ---------------
    user_history: dict[str, Any] = user_history_map.get(user_id, {})
    if not user_history:
        logger.debug(
            "No history found for user_id=%s; using empty history.", user_id
        )

    # ---- Stage 1: Claim analysis ------------------------------------------
    try:
        claim_result: dict[str, Any] = analyze_claim(
            user_claim=user_claim,
            claim_object=claim_object,
            user_history=user_history,
        )
    except Exception as exc:
        logger.warning(
            "claim_engine.analyze_claim failed for user_id=%s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        claim_result = {
            "claimed_issue": "unknown",
            "claimed_part": "unknown",
            "prompt_injection": False,
            "history_risk": False,
        }

    # ---- Stage 2: Image / evidence analysis -------------------------------
    # Rebase relative paths onto dataset_dir so evidence_engine.Path.resolve()
    # produces the correct absolute path (claims.csv paths are relative to dataset/).
    absolute_image_paths: str = _rebase_image_paths(image_paths, dataset_dir)
    try:
        evidence_result: dict[str, Any] = analyze_images(
            image_paths=absolute_image_paths,
            claim_object=claim_object,
            claim_result=claim_result,
        )
    except Exception as exc:
        logger.warning(
            "evidence_engine.analyze_images failed for user_id=%s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        evidence_result = {
            "valid_image": False,
            "issue_type": "unknown",
            "object_part": "unknown",
            "severity": "unknown",
            "risk_flags": ["damage_not_visible"],
            "supporting_image_ids": [],
        }

    # ---- Stage 3: Final review / decision ---------------------------------
    try:
        review_result: dict[str, Any] = review_claim(
            claim_result=claim_result,
            evidence_result=evidence_result,
            evidence_requirements=evidence_requirements,
            claim_object=claim_object,
        )
    except Exception as exc:
        logger.warning(
            "review_engine.review_claim failed for user_id=%s: %s",
            user_id,
            exc,
            exc_info=True,
        )
        review_result = {
            "evidence_standard_met": False,
            "evidence_standard_met_reason": "Processing error; manual review required",
            "risk_flags": "manual_review_required",
            "issue_type": evidence_result.get("issue_type", "unknown"),
            "object_part": evidence_result.get("object_part", "unknown"),
            "claim_status": "not_enough_information",
            "claim_status_justification": "Pipeline error; claim could not be evaluated",
            "supporting_image_ids": "none",
            "valid_image": evidence_result.get("valid_image", False),
            "severity": evidence_result.get("severity", "unknown"),
        }

    # ---- Assemble output row ----------------------------------------------
    output_row: dict[str, Any] = {
        # Pass-through input fields
        "user_id": user_id,
        "image_paths": image_paths,
        "user_claim": user_claim,
        "claim_object": claim_object,
        # Derived output fields from review_result
        "evidence_standard_met": review_result.get("evidence_standard_met", False),
        "evidence_standard_met_reason": review_result.get(
            "evidence_standard_met_reason", ""
        ),
        "risk_flags": review_result.get("risk_flags", "none"),
        "issue_type": review_result.get("issue_type", "unknown"),
        "object_part": review_result.get("object_part", "unknown"),
        "claim_status": review_result.get("claim_status", "not_enough_information"),
        "claim_status_justification": review_result.get(
            "claim_status_justification", ""
        ),
        "supporting_image_ids": review_result.get("supporting_image_ids", "none"),
        "valid_image": review_result.get("valid_image", False),
        "severity": review_result.get("severity", "unknown"),
    }

    return output_row


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------

def run_pipeline() -> None:
    """
    Execute the full claim verification pipeline.

    Steps:
        1. Load claims, user history, and evidence requirements.
        2. Process each claim row through the three analysis engines.
        3. Collect output rows.
        4. Write output.csv with the exact schema defined in OUTPUT_COLUMNS.

    Exits with code 1 on fatal errors (dataset not found, missing columns,
    output write failure).
    """
    start_time = time.monotonic()
    logger.info("=== Insurance Claim Verification Pipeline starting ===")

    # ---- Step 1: Load datasets --------------------------------------------
    logger.info("Loading datasets...")

    try:
        claims_df: pd.DataFrame = load_claims()
        logger.info("Loaded %d claim(s) from claims.csv", len(claims_df))
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load claims.csv: %s", exc)
        sys.exit(1)

    try:
        user_history_map: dict[str, dict[str, Any]] = load_user_history()
        logger.info(
            "Loaded history for %d user(s) from user_history.csv",
            len(user_history_map),
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load user_history.csv: %s", exc)
        sys.exit(1)

    try:
        evidence_requirements: dict[str, dict[str, Any]] = load_evidence_requirements()
        logger.info(
            "Loaded %d evidence requirement(s) from evidence_requirements.csv",
            len(evidence_requirements),
        )
    except (FileNotFoundError, ValueError) as exc:
        logger.error("Failed to load evidence_requirements.csv: %s", exc)
        sys.exit(1)

    # ---- Step 2: Process each claim row -----------------------------------
    output_rows: list[dict[str, Any]] = []
    errors: list[tuple[int, str]] = []  # (row_index, error_message)

    # Resolve dataset/ directory once; used to rebase relative image paths.
    dataset_dir: Path = create_dataset_paths()["claims"].parent

    logger.info("Processing %d claim(s)...", len(claims_df))

    for idx, row in claims_df.iterrows():
        row_num = int(idx) + 1  # type: ignore[arg-type]
        user_id_label = str(row.get("user_id", f"row_{row_num}"))

        try:
            output_row = process_claim_row(
                row=row,
                user_history_map=user_history_map,
                evidence_requirements=evidence_requirements,
                dataset_dir=dataset_dir,
            )
            output_rows.append(output_row)
            logger.debug("Row %d (user_id=%s): OK", row_num, user_id_label)

        except Exception as exc:
            # Last-resort catch: log and produce a minimal fallback row
            error_msg = f"Unhandled exception at row {row_num} (user_id={user_id_label}): {exc}"
            logger.error(error_msg, exc_info=True)
            errors.append((row_num, str(exc)))

            # Emit a fallback row so output row count matches input row count
            fallback_row: dict[str, Any] = {
                "user_id": str(row.get("user_id", "")),
                "image_paths": str(row.get("image_paths", "")),
                "user_claim": str(row.get("user_claim", "")),
                "claim_object": str(row.get("claim_object", "")),
                "evidence_standard_met": False,
                "evidence_standard_met_reason": "Unhandled pipeline error",
                "risk_flags": "manual_review_required",
                "issue_type": "unknown",
                "object_part": "unknown",
                "claim_status": "not_enough_information",
                "claim_status_justification": "Unhandled pipeline error; manual review required",
                "supporting_image_ids": "none",
                "valid_image": False,
                "severity": "unknown",
            }
            output_rows.append(fallback_row)

    logger.info("Processed %d claim(s); %d error(s) encountered.", len(output_rows), len(errors))

    # ---- Step 3: Write output.csv -----------------------------------------
    paths = create_dataset_paths()
    output_path: Path = paths["output"]

    try:
        output_df = pd.DataFrame(output_rows, columns=OUTPUT_COLUMNS)
        output_df.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL)
        logger.info("Results written to %s (%d row(s))", output_path, len(output_df))
    except Exception as exc:
        logger.error("Failed to write output.csv: %s", exc, exc_info=True)
        sys.exit(1)

    # ---- Summary ----------------------------------------------------------
    elapsed = time.monotonic() - start_time
    logger.info(
        "=== Pipeline complete in %.2fs | %d claims processed | %d error(s) ===",
        elapsed,
        len(output_rows),
        len(errors),
    )

    if errors:
        logger.warning(
            "%d row(s) encountered errors and received fallback output:", len(errors)
        )
        for row_num, msg in errors:
            logger.warning("  Row %d: %s", row_num, msg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_pipeline()
