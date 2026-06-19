"""
Review Engine Module

This module handles final claim decision logic by synthesizing:
- Claim analysis (issue/part claims, prompt injection detection)
- Evidence analysis (image validity, damage detection, severity)
- User history risk assessment
- Evidence requirements validation

It applies deterministic rule-based logic to produce:
- Evidence standard assessment
- Claim status determination
- Combined risk flags
- Final decision package for CSV output

No AI model calls. All logic is deterministic and rule-based.

Architecture:
- Helper functions for comparisons and formatting
- Separated concerns: evidence evaluation, status determination, risk aggregation
- Clean decision tree for claim status logic
- CSV-compatible output formatting

Python 3.11+
"""

from typing import Dict, Any, Tuple, List


# ============================================================================
# CONSTANTS
# ============================================================================

# Allowed claim statuses
CLAIM_STATUSES = {"supported", "contradicted", "not_enough_information"}


# ============================================================================
# FORMATTING HELPERS
# ============================================================================

def _format_list_to_string(items: List[str]) -> str:
    """
    Convert list of strings to semicolon-separated format for CSV output.
    
    Returns "none" if list is empty or None.
    
    Args:
        items: List of strings to format
        
    Returns:
        Semicolon-separated string or "none"
    """
    if not items or len(items) == 0:
        return "none"
    return ";".join(str(item) for item in items if item)


# ============================================================================
# COMPARISON HELPERS
# ============================================================================

def _issues_match(claimed: str, detected: str) -> bool:
    """
    Determine if claimed and detected issues match (case-insensitive).
    
    Args:
        claimed: Issue claimed by user
        detected: Issue detected in evidence
        
    Returns:
        True if issues match, False otherwise
    """
    if not claimed or not detected:
        return False
    return claimed.lower().strip() == detected.lower().strip()


def _parts_match(claimed: str, detected: str) -> bool:
    """
    Determine if claimed and detected parts match (case-insensitive).
    
    Args:
        claimed: Part claimed by user
        detected: Part detected in evidence
        
    Returns:
        True if parts match, False otherwise
    """
    if not claimed or not detected:
        return False
    return claimed.lower().strip() == detected.lower().strip()


# ============================================================================
# PRIMARY DECISION FUNCTIONS
# ============================================================================

def evaluate_evidence_standard(
    claim_object: str,
    claimed_issue: str,
    evidence_requirements: Dict[str, Any],
    evidence_result: Dict[str, Any]
) -> Tuple[bool, str]:
    """
    Determine whether minimum evidence requirements are satisfied.
    
    Decision criteria:
    1. Images must be valid
    2. Damage must be visible (not flagged as damage_not_visible)
    3. Image quality must be acceptable (not blurry, cropped, wrong object, etc.)
    4. Minimum image count requirement must be met
    
    Args:
        claim_object: Type of object ("car", "laptop", "package")
        claimed_issue: Type of damage claimed (e.g., "dent", "scratch")
        evidence_requirements: Nested dict structure:
            {
                "car": {
                    "dent": {"min_images": 2},
                    "scratch": {"min_images": 1},
                    ...
                },
                "laptop": {...},
                "package": {...}
            }
            If not nested or missing, minimum requirement is 1 image.
        evidence_result: Dict from evidence_engine.analyze_images() with keys:
            valid_image, issue_type, object_part, severity, risk_flags, supporting_image_ids
        
    Returns:
        Tuple of (standard_met: bool, reason: str)
        
    Examples:
        (True, "Required evidence is visible")
        (False, "Damage area not clearly visible")
        (False, "Insufficient images: 0 provided, 2 required")
    """
    # Check 1: Images must be valid
    if not evidence_result.get("valid_image", False):
        return (False, "No valid images provided")
    
    # Check 2: Extract risk flags for quality assessment
    risk_flags = set(evidence_result.get("risk_flags", []))
    
    # Check 3: Critical image quality issues that prevent assessment
    critical_quality_issues = {
        "blurry_image": "Image is too blurry to assess damage",
        "cropped_or_obstructed": "Damage area cropped or obstructed",
        "wrong_object": "Image does not show the claimed object",
        "damage_not_visible": "Damage area not clearly visible",
    }
    
    for flag, reason in critical_quality_issues.items():
        if flag in risk_flags:
            return (False, reason)
    
    # Check 4: Minimum image count requirement
    supporting_ids = evidence_result.get("supporting_image_ids", [])
    min_images_required = 1  # Default
    
    # Look up specific requirement if available
    # TODO:
    # Adapt evidence requirement lookup after inspecting
    # dataset/evidence_requirements.csv structure.
    if isinstance(evidence_requirements, dict):
        object_reqs = evidence_requirements.get(claim_object.lower(), {})
        issue_reqs = object_reqs.get(claimed_issue.lower(), {})
        min_images_required = issue_reqs.get("min_images", 1)
    
    image_count = len(supporting_ids) if supporting_ids else 0
    if image_count < min_images_required:
        return (
            False,
            f"Insufficient images: {image_count} provided, {min_images_required} required"
        )
    
    # All checks passed
    return (True, "Required evidence is visible")


def determine_claim_status(
    claim_result: Dict[str, Any],
    evidence_result: Dict[str, Any]
) -> Tuple[str, str]:
    """
    Determine final claim status by comparing claim against evidence.
    
    Decision tree:
    1. If no valid images OR issue not detected → not_enough_information
    2. If severity is "none" → contradicted (no damage found)
    3. If claimed issue AND part match detected → supported
    4. If issue OR part mismatch → contradicted
    5. Otherwise → not_enough_information
    
    Args:
        claim_result: Dict from claim_engine.analyze_claim() with keys:
            claimed_issue, claimed_part, prompt_injection, history_risk
        evidence_result: Dict from evidence_engine.analyze_images() with keys:
            valid_image, issue_type, object_part, severity, risk_flags, supporting_image_ids
        
    Returns:
        Tuple of (status: str, justification: str)
        Status is one of: "supported", "contradicted", "not_enough_information"
        
    Examples:
        ("supported", "Claimed damage and location confirmed in visual evidence")
        ("contradicted", "Evidence shows no visible damage on the claimed object")
        ("not_enough_information", "Insufficient visual evidence to determine claim status")
    """
    # Extract key values
    valid_image = evidence_result.get("valid_image", False)
    detected_issue = evidence_result.get("issue_type", "unknown")
    detected_part = evidence_result.get("object_part", "unknown")
    severity = evidence_result.get("severity", "unknown")
    
    claimed_issue = claim_result.get("claimed_issue", "unknown")
    claimed_part = claim_result.get("claimed_part", "unknown")
    
    # TODO:
    # Future versions should handle partial matches and
    # unknown claimed parts more gracefully instead of
    # immediately treating them as contradictions.
    
    # Rule 1: No valid images or issue not detected
    if not valid_image or detected_issue == "unknown":
        return (
            "not_enough_information",
            "Insufficient visual evidence to determine claim status"
        )
    
    # Rule 2: Damage not present (severity = "none")
    if severity == "none":
        return (
            "contradicted",
            "Evidence shows no visible damage on the claimed object"
        )
    
    # Rule 3: Both issue and part match
    if _issues_match(claimed_issue, detected_issue) and _parts_match(claimed_part, detected_part):
        return (
            "supported",
            "Claimed damage and location confirmed in visual evidence"
        )
    
    # Rule 4a: Issue mismatch
    if not _issues_match(claimed_issue, detected_issue):
        return (
            "contradicted",
            f"Detected damage type '{detected_issue}' does not match claim '{claimed_issue}'"
        )
    
    # Rule 4b: Part mismatch
    if not _parts_match(claimed_part, detected_part):
        return (
            "contradicted",
            f"Damage detected on '{detected_part}' but claim specifies '{claimed_part}'"
        )
    
    # Rule 5: Fallback (should rarely reach here)
    return (
        "not_enough_information",
        "Cannot determine claim status from available evidence"
    )


def build_risk_flags(
    claim_result: Dict[str, Any],
    evidence_result: Dict[str, Any]
) -> List[str]:
    """
    Combine risk flags from claim and evidence analysis.
    
    Aggregates:
    - "text_instruction_present" if prompt injection detected in claim
    - "user_history_risk" if user history is risky
    - All risk flags from evidence analysis
    
    Removes duplicates and returns sorted list for deterministic output.
    
    Args:
        claim_result: Dict from claim_engine.analyze_claim() with keys:
            claimed_issue, claimed_part, prompt_injection, history_risk
        evidence_result: Dict from evidence_engine.analyze_images() with keys:
            valid_image, issue_type, object_part, severity, risk_flags, supporting_image_ids
        
    Returns:
        Sorted list of unique risk flags
        
    Example:
        ["text_instruction_present", "user_history_risk", "blurry_image"]
    """
    risk_flags_set = set()
    
    # Add text instruction flag if prompt injection detected
    if claim_result.get("prompt_injection", False):
        risk_flags_set.add("text_instruction_present")
    
    # Add user history risk flag if detected
    if claim_result.get("history_risk", False):
        risk_flags_set.add("user_history_risk")
    
    # Add all evidence-based risk flags
    evidence_flags = evidence_result.get("risk_flags", [])
    if evidence_flags:
        risk_flags_set.update(evidence_flags)
    
    # Return as sorted list for deterministic output
    return sorted(list(risk_flags_set))


def review_claim(
    claim_result: Dict[str, Any],
    evidence_result: Dict[str, Any],
    evidence_requirements: Dict[str, Any],
    claim_object: str
) -> Dict[str, Any]:
    """
    Main orchestrator: synthesize claim and evidence into final decision.
    
    Workflow:
    1. Call evaluate_evidence_standard() to check evidence quality
    2. Call determine_claim_status() to assess claim credibility
    3. Call build_risk_flags() to aggregate all risk factors
    4. Format outputs for CSV compatibility (lists → semicolon-separated strings)
    5. Assemble final decision package
    
    Args:
        claim_result: Dict from claim_engine.analyze_claim() with keys:
            claimed_issue (str), claimed_part (str),
            prompt_injection (bool), history_risk (bool)
        evidence_result: Dict from evidence_engine.analyze_images() with keys:
            valid_image (bool), issue_type (str), object_part (str),
            severity (str), risk_flags (list[str]), supporting_image_ids (list[str])
        claim_object: Type of object being claimed ("car", "laptop", "package", or other)
        evidence_requirements: Dict with nested structure:
            {
                "car": {
                    "dent": {"min_images": 2},
                    "scratch": {"min_images": 1},
                    ...
                },
                "laptop": {...},
                "package": {...}
            }
        
    Returns:
        Dict with exact schema for CSV output:
        {
            "evidence_standard_met": bool,
            "evidence_standard_met_reason": str,
            "risk_flags": str,                    # semicolon-separated or "none"
            "issue_type": str,
            "object_part": str,
            "claim_status": str,
            "claim_status_justification": str,
            "supporting_image_ids": str,          # semicolon-separated or "none"
            "valid_image": bool,
            "severity": str
        }
        
    Example:
        {
            "evidence_standard_met": True,
            "evidence_standard_met_reason": "Required evidence is visible",
            "risk_flags": "user_history_risk;blurry_image",
            "issue_type": "dent",
            "object_part": "rear_bumper",
            "claim_status": "supported",
            "claim_status_justification": "Claimed damage and location confirmed in visual evidence",
            "supporting_image_ids": "case_001;case_002",
            "valid_image": True,
            "severity": "medium"
        }
    """

    evidence_standard_met, evidence_reason = evaluate_evidence_standard(
        claim_object=claim_object,
        claimed_issue=claim_result.get("claimed_issue", "unknown"),
        evidence_requirements=evidence_requirements,
        evidence_result=evidence_result
    )
    
    # Step 2: Determine claim status
    claim_status, status_justification = determine_claim_status(
        claim_result=claim_result,
        evidence_result=evidence_result
    )
    
    # Step 3: Build combined risk flags
    risk_flags = build_risk_flags(
        claim_result=claim_result,
        evidence_result=evidence_result
    )
    
    # Step 4: Format outputs for CSV (lists → semicolon-separated strings)
    risk_flags_str = _format_list_to_string(risk_flags)
    supporting_ids = evidence_result.get("supporting_image_ids", [])
    supporting_ids_str = _format_list_to_string(supporting_ids)
    
    # Step 5: Assemble final decision package with exact schema
    return {
        "evidence_standard_met": evidence_standard_met,
        "evidence_standard_met_reason": evidence_reason,
        "risk_flags": risk_flags_str,
        "issue_type": evidence_result.get("issue_type", "unknown"),
        "object_part": evidence_result.get("object_part", "unknown"),
        "claim_status": claim_status,
        "claim_status_justification": status_justification,
        "supporting_image_ids": supporting_ids_str,
        "valid_image": evidence_result.get("valid_image", False),
        "severity": evidence_result.get("severity", "unknown"),
    }
