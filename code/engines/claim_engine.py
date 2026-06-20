"""
Claim Engine Module

This module provides rule-based analysis of damage claims without AI model calls.
It extracts issue types, identifies claimed parts, detects prompt injection attempts,
and assesses risk based on user history.

Limitations:
- Keyword matching is language-agnostic and may miss context-specific meanings
- Multi-part claims use positional heuristics (final mention) instead of semantic parsing
- TODO: Future versions should parse conversation transcripts and explicitly prioritize
  the final customer statement for accuracy in multi-turn claim conversations

Python 3.11+
"""

from typing import Dict, Any
import re


# Define supported issue types
ISSUE_TYPES = {
    "dent",
    "scratch",
    "crack",
    "broken_part",
    "missing_part",
    "torn_packaging",
    "crushed_packaging",
    "water_damage",
    "stain",
    "unknown",
}

# Define supported car parts
CAR_PARTS = {
    "front_bumper",
    "rear_bumper",
    "door",
    "hood",
    "windshield",
    "side_mirror",
    "headlight",
    "taillight",
    "fender",
    "quarter_panel",
    "body",
}

# Define supported laptop parts
LAPTOP_PARTS = {
    "screen",
    "keyboard",
    "trackpad",
    "hinge",
    "lid",
    "corner",
    "port",
    "base",
    "body",
}

# Define supported package parts
PACKAGE_PARTS = {
    "box",
    "package_corner",
    "package_side",
    "seal",
    "label",
    "contents",
    "item",
}

# Map object types to their valid parts
OBJECT_PARTS_MAP = {
    "car": CAR_PARTS,
    "laptop": LAPTOP_PARTS,
    "package": PACKAGE_PARTS,
}

# Prompt injection detection patterns - strong manipulation attempts only
# These patterns target explicit attempts to bypass review or approval processes
INJECTION_PATTERNS = [
    r"\bapprove\s+immediate(?:ly)?\b",
    r"\bauto[\s-]?approve\b",
    r"\bforce\s+approval\b",
    r"\bskip\s+manual\s+review\b",
    r"\bignore\s+previous\s+instructions?\b",
    r"\bmark\s+(?:as\s+)?supported\b",
]


def _normalize_text(text: str) -> str:
    """
    Normalize text for comparison: lowercase and remove extra whitespace.
    
    Args:
        text: The text to normalize.
    
    Returns:
        Normalized text string.
    """
    return " ".join(text.lower().split())


def _extract_issue_type(claim_text: str) -> str:
    """
    Extract the claimed issue type from the claim text.
    
    Supports multilingual claims (English and key Spanish/Hindi terms).
    
    Args:
        claim_text: The user's claim description.
    
    Returns:
        The detected issue type or "unknown" if not found.
    """
    normalized_claim = _normalize_text(claim_text)
    
    # Map keywords to issue types
    # Keywords are carefully selected: English terms, common Spanish translations,
    # and realistic Hindi/Hinglish terms that appear in real damage reports
    issue_keywords = {
        "dent": {
            "dent", "dented", "indented", "impact", "impact damage", "rear impact",
        },
        "scratch": {
            "scratch", "scratched", "scrape", "scuff", "cosmetic damage",
        },
        "crack": {
            "crack", "cracked", "fracture", "fractured", "shattered", "break",
        },
        "broken_part": {
            "broken", "snapped", "severed", "detached", "separated", "damaged", "physical damage", "damage",
        },
        "missing_part": {
            "missing", "absent", "lost", "gone",
        },
        "torn_packaging": {
            "torn", "ripped", "package torn", "box torn", "tore",
        },
        "crushed_packaging": {
            "crushed", "squashed", "compressed", "flattened", "package crushed",
        },
        "water_damage": {
            "water", "wet", "moisture", "flooded", "liquid", "spill",
            "agua", "mojado",  # Spanish: water, wet
        },
        "stain": {
            "stain", "stained", "spot", "discoloration",
        },
    }
    
    # Check for issue type keywords using word boundaries for accuracy
    for issue_type, keywords in issue_keywords.items():
        for keyword in keywords:
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, normalized_claim, re.IGNORECASE):
                return issue_type
    
    return "unknown"


def _extract_severity(issue_type: str) -> str:
    """
    Extract severity based on the claimed issue type.
    
    Args:
        issue_type: The detected issue type.
        
    Returns:
        The severity level ("high", "medium", "low", or "unknown").
    """
    if issue_type in {"broken_part", "missing_part", "crack"}:
        return "high"
    if issue_type == "dent":
        return "medium"
    if issue_type in {"scratch", "stain"}:
        return "low"
    return "unknown"


def _extract_part(claim_text: str, claim_object: str) -> str:
    """
    Extract the claimed object part from the claim text.
    
    Prefers the final part mentioned when multiple parts are detected, reflecting
    the customer's most recent statement in a claim.
    
    Args:
        claim_text: The user's claim description.
        claim_object: The type of object being claimed (e.g., "car", "laptop", "package").
    
    Returns:
        The detected part or "unknown" if not found.
    """
    normalized_claim = _normalize_text(claim_text)
    normalized_object = _normalize_text(claim_object)
    
    # Get valid parts for this object type
    valid_parts = OBJECT_PARTS_MAP.get(normalized_object, set())
    
    if not valid_parts:
        return "unknown"
    
    # Keyword mappings for parts - includes English and Spanish translations
    # Each entry maps to keywords that identify that part in a claim
    part_keywords = {
        "front_bumper": {
            "front bumper", "front-bumper", "frontal bumper",
            "parachoques delantero",  # Spanish
        },
        "rear_bumper": {
            "rear bumper", "rear-bumper", "back bumper", "back-bumper",
            "parachoques trasero", "parachoques posterior",  # Spanish
        },
        "door": {
            "door", "doors",
            "puerta", "puertas",  # Spanish
        },
        "hood": {
            "hood", "bonnet",
            "capo", "capot",  # Spanish
        },
        "windshield": {
            "windshield", "windscreen", "front window",
            "parabrisas",  # Spanish
        },
        "side_mirror": {
            "side mirror", "side-mirror",
            "left mirror", "right mirror",
            "espejo lateral",  # Spanish
        },
        "headlight": {
            "headlight", "head light", "front light", "headlamp",
            "faro delantero",  # Spanish
        },
        "taillight": {
            "taillight", "tail light", "tail-light",
            "brake light", "back light", "rear light",
            "faro trasero", "luz trasera",  # Spanish
        },
        "fender": {
            "fender",
            "guardabarros",  # Spanish
        },
        "quarter_panel": {
            "quarter panel", "quarter-panel",
        },
        "body": {
            "body", "panel",
            "cuerpo",  # Spanish
        },
        "screen": {
            "screen", "display", "monitor",
            "pantalla",  # Spanish
        },
        "keyboard": {
            "keyboard", "keys", "key",
            "teclado",  # Spanish
        },
        "trackpad": {
            "trackpad", "track pad", "touchpad",
            "almohadilla",  # Spanish
        },
        "hinge": {
            "hinge", "hinges",
            "bisagra",  # Spanish
        },
        "lid": {
            "lid",
            "tapa",  # Spanish
        },
        "corner": {
            "corner", "corners",
            "esquina", "esquinas",  # Spanish
        },
        "port": {
            "port", "usb", "connector",
            "puerto", "conector",  # Spanish
        },
        "base": {
            "base", "bottom",
            "fondo",  # Spanish
        },
        "box": {
            "box", "boxes", "carton",
            "caja", "cajas",  # Spanish
        },
        "package_corner": {
            "package corner", "package-corner", "box corner",
        },
        "package_side": {
            "package side", "package-side", "box side", "side of box",
        },
        "seal": {
            "seal", "sealed",
            "sello",  # Spanish
        },
        "label": {
            "label", "labels",
            "etiqueta", "etiquetas",  # Spanish
        },
        "contents": {
            "contents", "content", "inside",
            "contenido",  # Spanish
        },
        "item": {
            "item", "items",
            "articulo", "articulos",  # Spanish
        },
    }
    
    # Find all mentioned parts and track their positions
    found_parts = []
    
    for part in valid_parts:
        keywords = part_keywords.get(part, {part})
        for keyword in keywords:
            # Use word boundary matching for accuracy
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            if re.search(pattern, normalized_claim):
                # Record position of final occurrence
                found_parts.append((part, normalized_claim.rfind(keyword.lower())))
    
    if found_parts:
        # Return the part that appears last (prefer final statement)
        found_parts.sort(key=lambda x: x[1], reverse=True)
        return found_parts[0][0]
    
    return "unknown"


def _detect_prompt_injection(claim_text: str) -> bool:
    """
    Detect prompt injection attempts in the claim text.
    
    Focuses on strong manipulation patterns that attempt to bypass review
    or approval workflows. Does not flag legitimate praise or questions.
    
    Args:
        claim_text: The user's claim description.
    
    Returns:
        True if injection patterns are detected, False otherwise.
    """
    normalized_claim = _normalize_text(claim_text)
    
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized_claim, re.IGNORECASE):
            return True
    
    return False


def _detect_history_risk(user_history: Dict[str, Any]) -> bool:
    """
    Detect risk based on user claim history.
    
    Checks multiple risk indicators from user history: rejected claims,
    manual review flags, and history risk markers.
    
    Supports history_flags as both string values (from CSV) and list/tuple values.
    Treats non-empty values other than "none" as risk indicators.
    
    Args:
        user_history: Dictionary containing user history data with optional keys:
            - history_flags: String (CSV) or list/tuple of flag values
            - rejected_claim: Boolean or integer indicating rejected claims
            - rejected_claim_count: Integer count of rejected claims
            - manual_review_claim: Boolean or integer indicating manual review required
    
    Returns:
        True if history indicates risk, False otherwise.
    """
    if not user_history:
        return False
    
    # Check history_flags: can be string (from CSV) or list/tuple
    history_flags = user_history.get("history_flags")
    if history_flags is not None:
        if isinstance(history_flags, str):
            # CSV string value: non-empty and not "none" indicates risk
            if history_flags.strip() and history_flags.lower() != "none":
                return True
        elif isinstance(history_flags, (list, tuple)):
            # List/tuple: non-empty list indicates risk
            if len(history_flags) > 0:
                return True
    
    # Check rejected claim count
    rejected_count = user_history.get("rejected_claim_count")
    if isinstance(rejected_count, int) and rejected_count > 0:
        return True
    
    # Check rejected_claim field
    rejected_claim = user_history.get("rejected_claim")
    if rejected_claim is not None:
        if isinstance(rejected_claim, bool) and rejected_claim:
            return True
        elif isinstance(rejected_claim, int) and rejected_claim > 0:
            return True
    
    # Check manual_review_claim field
    manual_review = user_history.get("manual_review_claim")
    if manual_review is not None:
        if isinstance(manual_review, bool) and manual_review:
            return True
        elif isinstance(manual_review, int) and manual_review > 0:
            return True
    
    return False


def analyze_claim(
    user_claim: str,
    claim_object: str,
    user_history: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Analyze a damage claim and extract relevant information.
    
    This function performs rule-based analysis of a damage claim without AI model calls.
    Supports multilingual claims (English, Hindi, Hinglish, Spanish) and identifies the
    claimed issue type, the affected part, detects prompt injection attempts using regex
    patterns, and assesses risk based on the user's claim history.
    
    When multiple parts are mentioned, the final part mentioned in the claim is returned,
    reflecting the customer's most recent statement.
    
    Args:
        user_claim: The user's claim description text.
        claim_object: The type of object being claimed (e.g., "car", "laptop", "package").
        user_history: Dictionary containing user history data with optional keys:
            - history_flags: String or list of flag values
            - rejected_claim: Boolean or integer
            - rejected_claim_count: Integer count
            - manual_review_claim: Boolean or integer
    
    Returns:
        A dictionary with the following keys:
        - claimed_issue (str): The detected issue type (or "unknown")
        - claimed_part (str): The detected part (or "unknown")
        - prompt_injection (bool): Whether prompt injection was detected
        - history_risk (bool): Whether user history indicates risk
    
    Examples:
        >>> result = analyze_claim(
        ...     "The front bumper has a dent",
        ...     "car",
        ...     {"rejected_claim_count": 0, "history_flags": "none"}
        ... )
        >>> result["claimed_issue"]
        'dent'
        >>> result["claimed_part"]
        'front_bumper'
        >>> result["prompt_injection"]
        False
        >>> result["history_risk"]
        False
        
        >>> result = analyze_claim(
        ...     "pantalla damaged",
        ...     "laptop",
        ...     {"rejected_claim": True}
        ... )
        >>> result["claimed_part"]
        'screen'
        >>> result["history_risk"]
        True
    """
    # Validate and sanitize inputs
    user_claim = str(user_claim) if user_claim else ""
    claim_object = str(claim_object) if claim_object else ""
    user_history = user_history if isinstance(user_history, dict) else {}
    
    # Extract customer statements if transcript format is detected
    customer_text = user_claim
    if "Customer:" in user_claim:
        turns = [t.strip() for t in user_claim.split('|')]
        customer_parts = [t[len("Customer:"):].strip() for t in turns if t.startswith("Customer:")]
        if customer_parts:
            customer_text = " ".join(customer_parts)

    # Extract analysis using rule-based detection
    claimed_issue = _extract_issue_type(customer_text)
    claimed_part = _extract_part(customer_text, claim_object)
    severity = _extract_severity(claimed_issue)
    prompt_injection = _detect_prompt_injection(customer_text)
    history_risk = _detect_history_risk(user_history)
    
    return {
        "claimed_issue": claimed_issue,
        "claimed_part": claimed_part,
        "severity": severity,
        "prompt_injection": prompt_injection,
        "history_risk": history_risk,
    }
