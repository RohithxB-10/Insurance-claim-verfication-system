import sys
from pathlib import Path
import time
import logging

_CODE_DIR = Path(__file__).resolve().parents[1]
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

from main import process_claim_row
from utils.csv_loader import (
    create_dataset_paths,
    load_sample_claims,
    load_evidence_requirements,
    load_user_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("evaluation.main")

def run_evaluation():
    start_time = time.monotonic()
    logger.info("=== Insurance Claim Evaluation Pipeline ===")
    
    try:
        sample_df = load_sample_claims()
        user_history_map = load_user_history()
        evidence_requirements = load_evidence_requirements()
    except Exception as exc:
        logger.error(f"Failed to load dataset: {exc}")
        sys.exit(1)
        
    dataset_dir = create_dataset_paths()["claims"].parent
    
    correct_statuses = 0
    total = len(sample_df)
    
    for idx, row in sample_df.iterrows():
        try:
            output_row = process_claim_row(row, user_history_map, evidence_requirements, dataset_dir)
            expected = str(row.get("claim_status", "")).strip()
            actual = output_row.get("claim_status", "")
            if expected == actual:
                correct_statuses += 1
        except Exception as e:
            logger.error(f"Error processing row {idx}: {e}")
            
    elapsed = time.monotonic() - start_time
    logger.info(f"Evaluated {total} sample claims in {elapsed:.2f}s")
    logger.info(f"Accuracy (claim_status): {correct_statuses}/{total} ({(correct_statuses/total)*100 if total > 0 else 0:.1f}%)")

if __name__ == "__main__":
    run_evaluation()
