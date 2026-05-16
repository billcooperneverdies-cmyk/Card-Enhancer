"""Validation Suite for Adaptive Preprocessor."""
import numpy as np
import cv2
import json
from scipy import stats
from typing import Dict, List, Optional
from .adaptive_preprocessor import FeatureExtractor


class ValidationSuite:
    """Comprehensive validation suite for adaptive preprocessing."""
    
    def __init__(self):
        self.synth_features: List[np.ndarray] = []
        self.ground_truth: Optional[np.ndarray] = None
        
    def generate_synthetic(self, n_samples: int = 500) -> Dict[str, List]:
        """Creates controlled test images with measurable features."""
        synth_imgs, gt_features = [], []
        for _ in range(n_samples):
            w, h = 1920, 1080
            bg_contrast = np.random.uniform(0.05, 0.4)
            glare_int = np.random.uniform(0, 0.3)
            lighting_uni = np.random.uniform(0.5, 1.0)
            finish = int(np.random.choice([0, 1], p=[0.7, 0.3]))
            
            base = np.ones((h, w), dtype=np.uint8) * 200
            if finish == 1:
                base = cv2.addWeighted(base, 1, np.random.normal(128, 40, (h,w)).astype(np.uint8), 0.3, 0)
                base[500:600, 800:900] = 255
                
            if glare_int > 0:
                base = np.clip(base * (1 + glare_int * np.random.uniform(0.5, 1.0, (h,w))), 0, 255).astype(np.uint8)
                
            synth_imgs.append(cv2.cvtColor(base, cv2.COLOR_GRAY2BGR))
            gt_features.append([float(finish), bg_contrast, w/4096, h/4096, lighting_uni, glare_int])
            
        self.synth_features = synth_imgs
        self.ground_truth = np.array(gt_features)
        return {"images": synth_imgs, "ground_truth": gt_features}
    
    def measure_extraction_accuracy(self) -> Dict:
        extractor = FeatureExtractor()
        errors = []
        for img, gt in zip(self.synth_features, self.ground_truth):
            pred = extractor.extract(img)
            errors.append(pred - gt)
            
        errors = np.array(errors)
        mae = np.mean(np.abs(errors), axis=0)
        ci_95 = stats.t.interval(0.95, len(errors)-1, loc=np.mean(errors, axis=0), scale=stats.sem(errors, axis=0))
        
        return {
            "mae": mae.tolist(),
            "confidence_intervals": ci_95,
            "pass_criteria_met": all(m < 0.05 for m in mae)
        }

    def analyze_real_world_distribution(self, real_images: List[np.ndarray]) -> Dict:
        feats = np.array([FeatureExtractor.extract(img) for img in real_images])
        stats_report = {}
        labels = ["finish", "contrast", "norm_w", "norm_h", "lighting", "glare", "aspect"]
        
        for i, name in enumerate(labels):
            col = feats[:, i]
            stats_report[name] = {
                "mean": float(col.mean()), 
                "std": float(col.std()),
                "q1": float(np.percentile(col, 25)), 
                "q3": float(np.percentile(col, 75)),
                "skew": float(stats.skew(col)), 
                "kurtosis": float(stats.kurtosis(col)),
                "outliers": int(np.sum(np.abs(col - col.mean()) > 3 * col.std()))
            }
        return stats_report

    def sensitivity_analysis(self, image: np.ndarray) -> Dict:
        base_feats = FeatureExtractor.extract(image)
        perturbations = {
            "gaussian_noise": lambda x: cv2.GaussianBlur(x + np.random.normal(0, 5, x.shape).astype(np.uint8), (3,3), 1),
            "brightness_shift": lambda x: np.clip(x + 20, 0, 255).astype(np.uint8),
            "jpeg_compression": lambda x: cv2.imdecode(cv2.imencode('.jpg', x, [cv2.IMWRITE_JPEG_QUALITY, 60])[1], cv2.IMREAD_COLOR),
        }
        sensitivity = {}
        
        for name, func in perturbations.items():
            try:
                perturbed = func(image.copy())
                pert_feats = FeatureExtractor.extract(perturbed)
                sensitivity[name] = np.abs(pert_feats - base_feats).tolist()
            except Exception:
                sensitivity[name] = None
            
        return sensitivity

    def correlation_redundancy(self, features_matrix: np.ndarray) -> Dict:
        corr = np.corrcoef(features_matrix, rowvar=False)
        labels = ["finish", "contrast", "w", "h", "lighting", "glare", "aspect"]
        pairs = []
        for i in range(corr.shape[0]):
            for j in range(i+1, corr.shape[1]):
                if abs(corr[i,j]) > 0.85:
                    pairs.append((labels[i], labels[j], float(corr[i,j])))
        return {"correlation_matrix": corr.tolist(), "highly_correlated": pairs}

    def export_audit_log(self, filepath: str = "validation_report.json", 
                         real_features: Optional[np.ndarray] = None):
        report = {
            "synthetic_accuracy": self.measure_accuracy() if self.ground_truth is not None else None,
            "feature_correlation": self.correlation_redundancy(real_features) if real_features is not None else None,
            "recommendations": []
        }
        if report["synthetic_accuracy"] and any(m > 0.05 for m in report["synthetic_accuracy"]["mae"]):
            report["recommendations"].append("Recalibrate feature extraction: MAE exceeds 0.05 threshold.")
        if report["feature_correlation"] and report["feature_correlation"]["highly_correlated"]:
            report["recommendations"].append("Reduce feature dimensionality: High correlations detected.")
            
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2)
        return report
