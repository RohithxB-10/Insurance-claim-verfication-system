"""
Evidence Engine Module (Multimodal VLM Edition)
"""

from pathlib import Path
from typing import Dict, Any, List
import os
import json
import base64
import urllib.request
import urllib.error
import logging

logger = logging.getLogger(__name__)

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
SUPPORTED_SEVERITY = {"none", "low", "medium", "high", "unknown"}
SUPPORTED_OBJECTS = {"car", "laptop", "package"}

def _normalize_image_path(image_path: str) -> Path:
    return Path(image_path).resolve()

def _is_valid_image_file(file_path: Path) -> bool:
    if not file_path.is_file(): return False
    if file_path.suffix.lower() not in VALID_EXTENSIONS: return False
    try:
        if not os.access(file_path, os.R_OK): return False
    except OSError: return False
    return True

def extract_image_ids(image_paths: str) -> List[str]:
    if not image_paths or not isinstance(image_paths, str): return []
    image_ids = []
    for path_str in image_paths.split(";"):
        path_str = path_str.strip()
        if not path_str: continue
        image_ids.append(Path(path_str).stem)
    return image_ids

def validate_images(image_paths: str) -> Dict[str, Any]:
    valid_files, missing_files, risk_flags = [], [], []
    if not image_paths or not isinstance(image_paths, str):
        return {"valid_image": False, "valid_files": [], "missing_files": [], "risk_flags": ["damage_not_visible"]}
    for path_str in image_paths.split(";"):
        path_str = path_str.strip()
        if not path_str: continue
        try:
            file_path = _normalize_image_path(path_str)
            if _is_valid_image_file(file_path): valid_files.append(str(file_path))
            else: missing_files.append(path_str)
        except OSError:
            missing_files.append(path_str)
    
    valid_image = len(valid_files) > 0 and len(missing_files) == 0
    if len(missing_files) > 0: risk_flags.append("damage_not_visible")
    
    return {
        "valid_image": valid_image,
        "valid_files": valid_files,
        "missing_files": missing_files,
        "risk_flags": risk_flags,
    }

def load_images(image_paths: str) -> List[str]:
    loaded_images = []
    if not image_paths or not isinstance(image_paths, str): return []
    for path_str in image_paths.split(";"):
        path_str = path_str.strip()
        if not path_str: continue
        try:
            file_path = _normalize_image_path(path_str)
            if _is_valid_image_file(file_path): loaded_images.append(str(file_path))
        except OSError: continue
    return loaded_images

def _encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def call_gemini_vision(images: List[str], claim_text: str, claim_object: str, api_key: str) -> dict:
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"
    prompt = f"You are an expert damage claim evaluator. Review the attached image(s) against the customer's claim: '{claim_text}'. The claimed object is a {claim_object}. Respond in strict JSON. Schema: {{\"issue_type\": \"dent|scratch|crack|broken_part|missing_part|torn_packaging|crushed_packaging|water_damage|stain|none|unknown\", \"object_part\": \"front_bumper|rear_bumper|door|hood|windshield|side_mirror|headlight|taillight|fender|body|screen|keyboard|trackpad|hinge|corner|box|seal|contents|unknown\", \"severity\": \"high|medium|low|none|unknown\", \"valid_image\": true|false, \"risk_flags\": [\"wrong_object\", \"wrong_angle\", \"damage_not_visible\", \"claim_mismatch\", \"blurry_image\"]}}. Only include applicable risk_flags. Set valid_image to false if the image is clearly NOT the claimed object type. If the damage contradicts the claim entirely (e.g. claim scratch but is destroyed, or wrong object part), include 'claim_mismatch'. If the image does not show the damaged area, include 'damage_not_visible'."
    
    parts = [{"text": prompt}]
    for img_path in images:
        parts.append({
            "inline_data": {
                "mime_type": "image/jpeg",
                "data": _encode_image(img_path)
            }
        })
        
    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"response_mime_type": "application/json", "temperature": 0.0}
    }
    
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))
        content = result["candidates"][0]["content"]["parts"][0]["text"]
        return json.loads(content)

def call_openai_vision(images: List[str], claim_text: str, claim_object: str, api_key: str) -> dict:
    url = "https://api.openai.com/v1/chat/completions"
    prompt = f"You are an expert damage claim evaluator. Review the attached image(s) against the customer's claim: '{claim_text}'. The claimed object is a {claim_object}. Respond in strict JSON. Schema: {{\"issue_type\": \"dent|scratch|crack|broken_part|missing_part|torn_packaging|crushed_packaging|water_damage|stain|none|unknown\", \"object_part\": \"front_bumper|rear_bumper|door|hood|windshield|side_mirror|headlight|taillight|fender|body|screen|keyboard|trackpad|hinge|corner|box|seal|contents|unknown\", \"severity\": \"high|medium|low|none|unknown\", \"valid_image\": true|false, \"risk_flags\": [\"wrong_object\", \"wrong_angle\", \"damage_not_visible\", \"claim_mismatch\", \"blurry_image\"]}}. Only include applicable risk_flags. Set valid_image to false if the image is clearly NOT the claimed object type. If the damage contradicts the claim entirely, include 'claim_mismatch'. If the image does not show the damaged area, include 'damage_not_visible'."
    
    content = [{"type": "text", "text": prompt}]
    for img_path in images:
        content.append({"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode_image(img_path)}"}})
        
    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 300,
        "temperature": 0.0,
        "response_format": {"type": "json_object"}
    }
    
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"}
    req = urllib.request.Request(url, data=json.dumps(payload).encode("utf-8"), headers=headers)
    with urllib.request.urlopen(req, timeout=15) as response:
        result = json.loads(response.read().decode("utf-8"))
        return json.loads(result["choices"][0]["message"]["content"])

def analyze_images(
    image_paths: str,
    claim_object: str,
    claim_result: Dict[str, Any] | None = None,
    user_claim: str = "",
) -> Dict[str, Any]:
    normalized_object = claim_object.strip().lower() if claim_object else ""
    risk_flags = []
    if normalized_object not in SUPPORTED_OBJECTS: risk_flags.append("manual_review_required")
    
    validation_result = validate_images(image_paths)
    all_image_ids = extract_image_ids(image_paths)
    
    valid_image_ids = []
    if validation_result["valid_files"]:
        valid_file_names = {Path(f).stem for f in validation_result["valid_files"]}
        valid_image_ids = [id for id in all_image_ids if id in valid_file_names]
    
    vlm_images = load_images(image_paths)
    combined_risk_flags = validation_result["risk_flags"] + risk_flags
    unique_risk_flags = list(dict.fromkeys(combined_risk_flags))
    
    claim_result = claim_result or {}
    base_result = {
        "valid_image": validation_result["valid_image"],
        "issue_type": claim_result.get("claimed_issue", "unknown"),
        "object_part": claim_result.get("claimed_part", "unknown"),
        "severity": claim_result.get("severity", "unknown"),
        "risk_flags": unique_risk_flags,
        "supporting_image_ids": valid_image_ids,
    }

    # Attempt VLM analysis
    if vlm_images:
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        openai_key = os.environ.get("OPENAI_API_KEY")
        
        try:
            vlm_data = None
            if gemini_key:
                logger.info("Using Gemini Vision API")
                vlm_data = call_gemini_vision(vlm_images, user_claim, normalized_object, gemini_key)
            elif openai_key:
                logger.info("Using OpenAI Vision API")
                vlm_data = call_openai_vision(vlm_images, user_claim, normalized_object, openai_key)
                
            if vlm_data:
                return {
                    "valid_image": vlm_data.get("valid_image", True),
                    "issue_type": vlm_data.get("issue_type", "unknown"),
                    "object_part": vlm_data.get("object_part", "unknown"),
                    "severity": vlm_data.get("severity", "unknown"),
                    "risk_flags": list(dict.fromkeys(unique_risk_flags + vlm_data.get("risk_flags", []))),
                    "supporting_image_ids": valid_image_ids,
                }
        except Exception as e:
            logger.error(f"VLM API Call Failed: {e}. Falling back to text deterministic mock.")

    return base_result
