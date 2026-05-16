"""Batch processing utilities for adaptive preprocessing."""
import concurrent.futures
from typing import List, Dict
import cv2
from .adaptive_preprocessor import AdaptivePreprocessor


def batch_process_cards(image_paths: List[str], preprocessor: AdaptivePreprocessor, 
                        max_workers: int = 4) -> List[Dict]:
    """Parallelized batch processing with ProcessPoolExecutor."""
    def _process_single(path: str) -> Dict:
        img = cv2.imread(path)
        if img is None:
            return {"path": path, "status": "load_failed", "error": "Could not read image"}
        result = preprocessor.process(img)
        result["path"] = path
        result["status"] = "success"
        return result
        
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(executor.map(_process_single, image_paths))
    return results
