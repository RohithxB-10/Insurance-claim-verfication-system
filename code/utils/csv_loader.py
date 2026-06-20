"""
CSV Data Loader Module

Provides utilities for loading and validating claim data from CSV files.
This module handles file I/O, data validation, and structure conversion
without performing any claim analysis or AI model calls.

Responsibilities:
- Locate dataset files using repository-relative paths
- Load CSV files and validate required columns
- Parse complex fields (e.g., history_flags, image_paths)
- Return structured data ready for downstream processing

Out of scope:
- Claim decision logic or evidence analysis
- Image processing or vision model integration
- Output file generation
- User history statistical analysis

Architecture:
- Centralized path management via create_dataset_paths()
- Dedicated loader functions for each CSV file type
- Clear, recoverable error handling with ValueError and FileNotFoundError
- Type-annotated for runtime safety and IDE support

Python 3.11+
"""

from pathlib import Path
from typing import Dict, Any, List
import pandas as pd


# ============================================================================
# CONSTANTS
# ============================================================================

# Required columns for each CSV file
REQUIRED_CLAIMS_COLUMNS = {"user_id", "image_paths", "user_claim", "claim_object"}
REQUIRED_SAMPLE_CLAIMS_COLUMNS = {
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
}
REQUIRED_USER_HISTORY_COLUMNS = {
    "user_id",
    "past_claim_count",
    "accept_claim",
    "manual_review_claim",
    "rejected_claim",
    "last_90_days_claim_count",
    "history_flags",
    "history_summary",
}
REQUIRED_EVIDENCE_REQUIREMENTS_COLUMNS = {
    "requirement_id",
    "claim_object",
    "applies_to",
    "minimum_image_evidence",
}



# ============================================================================
# PATH MANAGEMENT
# ============================================================================

def create_dataset_paths() -> Dict[str, Path]:
    """
    Create dataset file paths relative to repository root.
    
    Returns a dictionary of pathlib.Path objects for all required dataset files.
    Paths are resolved relative to the repository root (parent of code/ directory).
    
    Returns:
        Dict[str, Path]: Dictionary mapping file names to absolute Path objects.
            Keys: "claims", "sample_claims", "user_history", "evidence_requirements", "output"
            Example:
                {
                    "claims": Path("/repo/dataset/claims.csv"),
                    "sample_claims": Path("/repo/dataset/sample_claims.csv"),
                    ...
                }
    
    Raises:
        RuntimeError: If repository root cannot be determined.
    
    Example:
        >>> paths = create_dataset_paths()
        >>> claims_file = paths["claims"]
        >>> print(claims_file.exists())
        True
    """
    # Infer repository root from the location of this file
    # code/utils/csv_loader.py -> parents[0]=utils, parents[1]=code, parents[2]=repo_root
    current_file = Path(__file__).resolve()
    repo_root = current_file.parents[2]
    
    # Verify we're in the right place by checking for AGENTS.md
    agents_file = repo_root / "AGENTS.md"
    if not agents_file.exists():
        raise RuntimeError(
            f"Repository structure invalid: expected AGENTS.md at {repo_root}, "
            f"but file not found. Current file: {current_file}"
        )
    
    # Construct dataset paths
    dataset_dir = repo_root / "dataset"
    
    paths = {
        "claims": dataset_dir / "claims.csv",
        "sample_claims": dataset_dir / "sample_claims.csv",
        "user_history": dataset_dir / "user_history.csv",
        "evidence_requirements": dataset_dir / "evidence_requirements.csv",
        "output": dataset_dir / "output.csv",
    }
    
    return paths


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def _verify_file_exists(file_path: Path, file_type: str) -> None:
    """
    Verify that a CSV file exists at the given path.
    
    Args:
        file_path: Path to the file to check.
        file_type: Human-readable description of file (e.g., "claims").
    
    Raises:
        FileNotFoundError: If file does not exist.
    """
    if not file_path.exists():
        raise FileNotFoundError(
            f"Dataset file not found: {file_type} at {file_path}"
        )


def _verify_columns_exist(
    df: pd.DataFrame, required_columns: set, file_type: str
) -> None:
    """
    Verify that all required columns exist in a DataFrame.
    
    Strips whitespace from actual column names before comparison.
    
    Args:
        df: DataFrame to check.
        required_columns: Set of column names that must be present.
        file_type: Human-readable description of file (e.g., "claims").
    
    Raises:
        ValueError: If any required column is missing.
    """
    # Normalize column names (strip whitespace)
    actual_columns = {col.strip() for col in df.columns}
    
    missing_columns = required_columns - actual_columns
    if missing_columns:
        raise ValueError(
            f"Missing required columns in {file_type}: {missing_columns}. "
            f"Available columns: {actual_columns}"
        )


def _normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize column names by stripping whitespace.
    
    Args:
        df: DataFrame with potentially whitespace-padded column names.
    
    Returns:
        DataFrame with normalized column names.
    """
    df.columns = df.columns.str.strip()
    return df


def _parse_history_flags(flags_str: str) -> List[str]:
    """
    Parse history_flags field from CSV into a list of flag strings.
    
    The history_flags field is semicolon-separated. Special case: "none"
    maps to an empty list.
    
    Args:
        flags_str: String representation of flags (e.g., "user_history_risk;manual_review_required").
    
    Returns:
        List[str]: List of individual flags (empty list if flags_str is "none").
    
    Example:
        >>> _parse_history_flags("none")
        []
        >>> _parse_history_flags("user_history_risk")
        ["user_history_risk"]
        >>> _parse_history_flags("user_history_risk;manual_review_required")
        ["user_history_risk", "manual_review_required"]
    """
    if pd.isna(flags_str) or flags_str.strip() == "":
        return []
    
    flags_str = flags_str.strip()
    if flags_str.lower() == "none":
        return []
    
    # Split by semicolon and strip whitespace from each flag
    flags = [flag.strip() for flag in flags_str.split(";")]
    return [flag for flag in flags if flag]  # Remove empty strings


# ============================================================================
# CSV LOADERS
# ============================================================================

def load_claims() -> pd.DataFrame:
    """
    Load and validate the main claims dataset.
    
    Loads dataset/claims.csv and validates that all required columns are present.
    Normalizes column names by stripping whitespace.
    
    Required columns:
    - user_id
    - image_paths
    - user_claim
    - claim_object
    
    Returns:
        pd.DataFrame: Claims dataframe with normalized column names.
                      Each row represents one claim to be evaluated.
    
    Raises:
        FileNotFoundError: If claims.csv does not exist.
        ValueError: If required columns are missing.
    
    Example:
        >>> df = load_claims()
        >>> print(df.shape)
        (100, 4)
        >>> print(df["claim_object"].unique())
        ['car' 'laptop' 'package']
    """
    paths = create_dataset_paths()
    file_path = paths["claims"]
    
    _verify_file_exists(file_path, "claims")
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Normalize column names
    df = _normalize_column_names(df)
    
    # Verify required columns
    _verify_columns_exist(df, REQUIRED_CLAIMS_COLUMNS, "claims")
    
    return df


def load_sample_claims() -> pd.DataFrame:
    """
    Load and validate the labeled sample claims dataset.
    
    Loads dataset/sample_claims.csv and validates that all required columns are present.
    Normalizes column names by stripping whitespace. Sample claims include expected outputs
    for development and evaluation.
    
    Required columns:
    - user_id, image_paths, user_claim, claim_object (input fields)
    - evidence_standard_met, evidence_standard_met_reason, risk_flags,
      issue_type, object_part, claim_status, claim_status_justification,
      supporting_image_ids, valid_image, severity (output fields)
    
    Returns:
        pd.DataFrame: Sample claims dataframe with all input and output columns.
                      Each row represents a labeled claim with expected outputs.
    
    Raises:
        FileNotFoundError: If sample_claims.csv does not exist.
        ValueError: If required columns are missing.
    
    Example:
        >>> df = load_sample_claims()
        >>> print(df["claim_status"].value_counts())
        supported    15
        not_enough_information    5
    """
    paths = create_dataset_paths()
    file_path = paths["sample_claims"]
    
    _verify_file_exists(file_path, "sample_claims")
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Normalize column names
    df = _normalize_column_names(df)
    
    # Verify required columns
    _verify_columns_exist(df, REQUIRED_SAMPLE_CLAIMS_COLUMNS, "sample_claims")
    
    return df


def load_user_history() -> Dict[str, Dict[str, Any]]:
    """
    Load and parse user historical data.
    
    Loads dataset/user_history.csv and converts it into a nested dictionary
    keyed by user_id. Parses the history_flags field into a list of individual flags.
    
    Returned structure:
        {
            "user_001": {
                "past_claim_count": 5,
                "accept_claim": 4,
                "manual_review_claim": 1,
                "rejected_claim": 0,
                "last_90_days_claim_count": 2,
                "history_flags": ["user_history_risk"],
                "history_summary": "...",
            },
            ...
        }
    
    Returns:
        Dict[str, Dict[str, Any]]: User history indexed by user_id.
                                   Each entry contains claim counts, flags list, and summary.
    
    Raises:
        FileNotFoundError: If user_history.csv does not exist.
        ValueError: If required columns are missing.
    
    Example:
        >>> history = load_user_history()
        >>> user_data = history["user_001"]
        >>> print(user_data["past_claim_count"])
        5
        >>> print(user_data["history_flags"])
        ["user_history_risk"]
    """
    paths = create_dataset_paths()
    file_path = paths["user_history"]
    
    _verify_file_exists(file_path, "user_history")
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Normalize column names
    df = _normalize_column_names(df)
    
    # Verify required columns
    _verify_columns_exist(df, REQUIRED_USER_HISTORY_COLUMNS, "user_history")
    
    # Build nested dictionary keyed by user_id
    user_history_dict: Dict[str, Dict[str, Any]] = {}
    
    for _, row in df.iterrows():
        user_id = str(row["user_id"])
        
        # Parse history_flags into a list
        history_flags = _parse_history_flags(str(row["history_flags"]))
        
        user_history_dict[user_id] = {
            "past_claim_count": int(row["past_claim_count"]),
            "accept_claim": int(row["accept_claim"]),
            "manual_review_claim": int(row["manual_review_claim"]),
            "rejected_claim": int(row["rejected_claim"]),
            "last_90_days_claim_count": int(row["last_90_days_claim_count"]),
            "history_flags": history_flags,
            "history_summary": str(row["history_summary"]),
        }
    
    return user_history_dict


def load_evidence_requirements() -> Dict[str, Dict[str, Any]]:
    """
    Load and structure the evidence requirements ruleset.
    
    Loads dataset/evidence_requirements.csv and converts it into a dictionary
    keyed by requirement_id. Preserves all fields for downstream reference.
    
    Returned structure:
        {
            "REQ_GENERAL_OBJECT_PART": {
                "requirement_id": "REQ_GENERAL_OBJECT_PART",
                "claim_object": "all",
                "applies_to": "general claim review",
                "minimum_image_evidence": "...",
            },
            ...
        }
    
    Returns:
        Dict[str, Dict[str, Any]]: Evidence requirements indexed by requirement_id.
                                   Each entry contains all requirement fields.
    
    Raises:
        FileNotFoundError: If evidence_requirements.csv does not exist.
        ValueError: If required columns are missing.
    
    Example:
        >>> reqs = load_evidence_requirements()
        >>> req = reqs["REQ_CAR_BODY_PANEL"]
        >>> print(req["applies_to"])
        "dent or scratch"
    """
    paths = create_dataset_paths()
    file_path = paths["evidence_requirements"]
    
    _verify_file_exists(file_path, "evidence_requirements")
    
    # Load CSV
    df = pd.read_csv(file_path)
    
    # Normalize column names
    df = _normalize_column_names(df)
    
    # Verify required columns
    _verify_columns_exist(
        df, REQUIRED_EVIDENCE_REQUIREMENTS_COLUMNS, "evidence_requirements"
    )
    
    # Build dictionary keyed by requirement_id
    evidence_dict: Dict[str, Dict[str, Any]] = {}
    
    for _, row in df.iterrows():
        requirement_id = str(row["requirement_id"])
        
        evidence_dict[requirement_id] = {
            "requirement_id": requirement_id,
            "claim_object": str(row["claim_object"]),
            "applies_to": str(row["applies_to"]),
            "minimum_image_evidence": str(row["minimum_image_evidence"]),
        }
    
    return evidence_dict
