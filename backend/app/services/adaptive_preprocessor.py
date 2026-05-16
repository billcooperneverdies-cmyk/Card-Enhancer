"""
Adaptive Preprocessor for Sports Card Enhancement.
Replaces hardcoded thresholds with ML-driven parameter selection.
Performance targets: Feature extraction <10ms, Full pipeline 8-12ms.
"""
import cv2
import numpy as np
import os
import time
import joblib
from typing import Dict, Any, List, Tuple, Optional
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from dataclasses import dataclass, asdict

try:
    import onnxruntime as ort
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False


@dataclass
class ParameterSet:
    """Container for adaptive preprocessing parameters."""
    canny_low: int
    canny_high: int
    blur_kernel_size: int
    morph_op: int  # 0:OPEN, 1:CLOSE, 2:DILATE, 3:ERODE
    poly_epsilon: float


class FeatureExtractor:
    """Extracts scale-invariant image features in <10ms on 1080p inputs."""
    TARGET_RES = (640, 480)
    
    @staticmethod
    def extract(image: np.ndarray) -> np.ndarray:
        h, w = image.shape[:2]
        
        # Fast downscale for speed; features are mathematically scale-invariant
        if w > FeatureExtractor.TARGET_RES[0] or h > FeatureExtractor.TARGET_RES[1]:
            img = cv2.resize(image, FeatureExtractor.TARGET_RES, interpolation=cv2.INTER_AREA)
        else:
            img = image.copy()
            
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_chan = lab[:, :, 0].astype(np.float32)
        
        # 1. Finish Type: Matte(0) vs Foil/Holographic(1)
        bright_mask = l_chan > 235
        bright_ratio = float(bright_mask.mean())
        lap_var = float(cv2.Laplacian(l_chan.astype(np.float64), cv2.CV_64F).var())
        finish = float(bright_ratio > 0.04 and lap_var > 350)
        
        # 2. Background Contrast: RMS contrast normalized to [0,1]
        rms_contrast = float(np.sqrt(np.mean((l_chan - l_chan.mean()) ** 2)) / 255.0)
        
        # 3. Input Dimensions (normalized to 4K)
        norm_w = float(w / 4096.0)
        norm_h = float(h / 4096.0)
        
        # 4. Lighting Uniformity: 1 - std of 4x4 grid means
        block_means = cv2.resize(l_chan, (4, 4), interpolation=cv2.INTER_NEAREST)
        lighting_uni = float(1.0 - (block_means.std() / 255.0))
        
        # 5. Surface Glare Intensity: % overexposed pixels
        glare_mask = l_chan > 248
        glare_intensity = float(glare_mask.mean())
        
        # 6. Aspect Ratio (normalized)
        aspect_ratio = float((w / h) / 2.0)
        
        return np.array([finish, rms_contrast, norm_w, norm_h, lighting_uni, glare_intensity, aspect_ratio], dtype=np.float32)


class ParameterClassifier:
    """Lightweight RandomForest for <0.2ms inference."""
    
    def __init__(self, model_path: str = "adaptive_params.pkl"):
        self.model_path = model_path
        self.model = MultiOutputRegressor(
            RandomForestRegressor(n_estimators=20, max_depth=4, random_state=42, n_jobs=1)
        )
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            
    def predict(self, features: np.ndarray) -> ParameterSet:
        pred = self.model.predict(features.reshape(1, -1))[0]
        return ParameterSet(
            canny_low=int(np.clip(pred[0], 10, 200)),
            canny_high=int(np.clip(pred[1], 30, 250)),
            blur_kernel_size=int(np.clip(np.round(pred[2]), 3, 11) // 2 * 2 + 1),
            morph_op=int(np.clip(np.round(pred[3]), 0, 3)),
            poly_epsilon=float(np.clip(pred[4], 0.01, 0.06))
        )
        
    def train(self, X: np.ndarray, y: np.ndarray):
        self.model.fit(X, y)
        joblib.dump(self.model, self.model_path)


def bootstrap_training_data(images: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Generates initial labels via parameter sweep + contour success scoring."""
    X, y = [], []
    for img in images:
        feats = FeatureExtractor.extract(img)
        best_score, best_params = -1, None
        
        # Sweep critical parameters
        for cl in [20, 40, 60, 80]:
            for ch in [60, 100, 140]:
                for eps in [0.015, 0.025, 0.035]:
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 1.5), cl, ch)
                    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    valid = []
                    for c in contours:
                        approx = cv2.approxPolyDP(c, eps, True)
                        if cv2.isContourConvex(approx) and len(approx) == 4:
                            if cv2.contourArea(c) > 0.05 * img.shape[0] * img.shape[1]:
                                valid.append(c)
                            
                    score = max([cv2.contourArea(v) for v in valid], default=0) * len(valid)
                    if score > best_score:
                        best_score, best_params = score, [cl, ch, 5.0, 0.0, eps]
        if best_score > 0:
            X.append(feats)
            y.append(best_params)
    return np.array(X), np.array(y)


class AdaptivePreprocessor:
    """Main adaptive preprocessing pipeline with confidence scoring and CNN fallback."""
    
    def __init__(self, model_path: str = "adaptive_params.pkl", 
                 cnn_fallback_path: Optional[str] = None):
        self.classifier = ParameterClassifier(model_path)
        self.metadata_log: List[Dict] = []
        
        self.cnn_session = None
        if cnn_fallback_path and os.path.exists(cnn_fallback_path) and ONNX_AVAILABLE:
            self.cnn_session = ort.InferenceSession(cnn_fallback_path, providers=['CPUExecutionProvider'])
            
    def process(self, image: np.ndarray, pass_num: int = 1) -> Dict[str, Any]:
        t0 = time.perf_counter()
        features = FeatureExtractor.extract(image)
        params = self.classifier.predict(features)
        
        # Apply OpenCV preprocessing
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (params.blur_kernel_size, params.blur_kernel_size), 1.5)
        edges = cv2.Canny(blurred, int(params.canny_low), int(params.canny_high))
        
        morph_ops = {0: cv2.MORPH_OPEN, 1: cv2.MORPH_CLOSE, 2: cv2.MORPH_DILATE, 3: cv2.MORPH_ERODE}
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edges = cv2.morphologyEx(edges, morph_ops[int(params.morph_op)], kernel)
        
        # Contour validation & confidence scoring
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contour, confidence = self._validate_contours(contours, image, params.poly_epsilon)
        
        # CNN fallback if confidence low
        if self.cnn_session and confidence < 0.75 and contour is not None:
            confidence = max(confidence, self._cnn_validate_contour(image, contour))
            
        elapsed = time.perf_counter() - t0
        record = {
            "timestamp": time.time(),
            "pass_num": pass_num,
            "features": features.tolist(),
            "params": asdict(params),
            "confidence": float(confidence),
            "processing_time_ms": elapsed * 1000,
            "contour_area": cv2.contourArea(contour) if contour is not None else 0,
            "cnn_fallback_used": self.cnn_session is not None and confidence < 0.8
        }
        self.metadata_log.append(record)
        
        return {"edges": edges, "contour": contour, "confidence": confidence, "metadata": record}
    
    def _validate_contours(self, contours: List[np.ndarray], original: np.ndarray, eps: float) -> Tuple[Optional[np.ndarray], float]:
        h, w = original.shape[:2]
        best, max_conf = None, 0.0
        for c in contours:
            area = cv2.contourArea(c)
            if not (0.05 * h * w < area < 0.95 * h * w):
                continue
            approx = cv2.approxPolyDP(c, eps, True)
            if len(approx) == 4 and cv2.isContourConvex(approx):
                corners = approx.reshape(4, 2)
                corner_sharpness = min([cv2.pointPolygonTest(c, tuple(p), True) for p in corners]) / -10.0
                rect_ratio = cv2.contourArea(approx) / (cv2.boundingRect(approx)[2] * cv2.boundingRect(approx)[3])
                conf = float(np.clip((rect_ratio * 0.6 + max(0, corner_sharpness) * 0.4), 0, 1))
                if conf > max_conf:
                    max_conf, best = conf, c
        return best, max_conf
    
    def _cnn_validate_contour(self, image: np.ndarray, contour: np.ndarray) -> float:
        if not ONNX_AVAILABLE or self.cnn_session is None:
            return 0.0
        x, y, bw, bh = cv2.boundingRect(contour)
        patch = cv2.resize(image[y:y+bh, x:x+bw], (224, 224), interpolation=cv2.INTER_LINEAR)
        inp = (patch.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        out = self.cnn_session.run(None, {self.cnn_session.get_inputs()[0].name: inp})
        return float(np.max(out[0][0]))
