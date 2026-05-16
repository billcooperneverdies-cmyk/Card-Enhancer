"""Blemish detection and removal service."""
import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from enum import Enum
import torch
import torch.nn as nn


class BlemishType(Enum):
    SCRATCH = "scratch"
    DUST = "dust"
    SCUFF = "scuff"
    PRINT_ARTIFACT = "print_artifact"
    BORDER_DAMAGE = "border_damage"
    HOLOGRAPHIC_DAMAGE = "holographic_damage"


@dataclass
class Blemish:
    """Detected blemish information."""
    type: BlemishType
    confidence: float
    bbox: Tuple[int, int, int, int]  # x, y, w, h
    severity: str
    mask: Optional[np.ndarray] = None


class BlemishDetector:
    """Detect various blemishes in sports card images."""
    
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        self.min_defect_size = int(10 + (1 - sensitivity) * 40)  # 10-50 pixels
        
    def detect_all(self, image: np.ndarray) -> List[Blemish]:
        """Detect all types of blemishes in the image."""
        blemishes = []
        
        # Detect each type
        blemishes.extend(self._detect_scratches(image))
        blemishes.extend(self._detect_dust(image))
        blemishes.extend(self._detect_scuffs(image))
        blemishes.extend(self._detect_print_artifacts(image))
        blemishes.extend(self._detect_border_damage(image))
        
        return blemishes
    
    def _detect_scratches(self, image: np.ndarray) -> List[Blemish]:
        """Detect scratches using line detection and morphological operations."""
        blemishes = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply CLAHE for better contrast
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        
        # Detect edges
        edges = cv2.Canny(enhanced, 50, 150)
        
        # Morphological operations to connect scratch segments
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Find lines using Hough transform
        lines = cv2.HoughLinesP(dilated, 1, np.pi/180, 
                                threshold=int(20 + (1-self.sensitivity)*30),
                                minLineLength=int(30 + (1-self.sensitivity)*50),
                                maxLineGap=10)
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate line properties
                length = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                angle = np.abs(np.arctan2(y2-y1, x2-x1) * 180 / np.pi)
                
                # Filter by angle (scratches are often diagonal or horizontal)
                if length > self.min_defect_size:
                    x = min(x1, x2)
                    y = min(y1, y2)
                    w = abs(x2 - x1) + 5  # Add padding
                    h = abs(y2 - y1) + 5
                    
                    confidence = min(1.0, length / 100) * self.sensitivity
                    severity = self._get_severity(confidence)
                    
                    blemishes.append(Blemish(
                        type=BlemishType.SCRATCH,
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        severity=severity
                    ))
        
        return blemishes
    
    def _detect_dust(self, image: np.ndarray) -> List[Blemish]:
        """Detect dust particles using blob detection."""
        blemishes = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Invert image (dust appears dark)
        inverted = 255 - gray
        
        # Threshold to find dark spots
        _, thresh = cv2.threshold(inverted, int(200 + self.sensitivity * 30), 255, 
                                  cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter by size
            if area < self.min_defect_size or area > 500:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate circularity
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter ** 2)
                
                # Dust particles are roughly circular
                if circularity > 0.5:
                    confidence = min(1.0, area / 100) * self.sensitivity
                    severity = self._get_severity(confidence)
                    
                    # Create mask for inpainting
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.drawContours(mask, [contour], -1, 255, -1)
                    
                    blemishes.append(Blemish(
                        type=BlemishType.DUST,
                        confidence=confidence,
                        bbox=(x, y, w, h),
                        severity=severity,
                        mask=mask[y:y+h, x:x+w]
                    ))
        
        return blemishes
    
    def _detect_scuffs(self, image: np.ndarray) -> List[Blemish]:
        """Detect surface scuffs using texture analysis."""
        blemishes = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Calculate local standard deviation (texture measure)
        mean, std_dev = cv2.meanStdDev(blurred)
        
        # Threshold high texture regions
        _, high_texture = cv2.threshold(std_dev[0][0] + blurred, 
                                        int(100 + self.sensitivity * 50), 255, 
                                        cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(high_texture, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_defect_size * 3:  # Scuffs are larger
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            # Calculate aspect ratio
            aspect_ratio = float(w) / h if h > 0 else 0
            
            # Scuffs are irregular in shape
            if 0.3 < aspect_ratio < 3.0:
                confidence = min(1.0, area / 500) * self.sensitivity
                severity = self._get_severity(confidence)
                
                blemishes.append(Blemish(
                    type=BlemishType.SCUFF,
                    confidence=confidence,
                    bbox=(x, y, w, h),
                    severity=severity
                ))
        
        return blemishes
    
    def _detect_print_artifacts(self, image: np.ndarray) -> List[Blemish]:
        """Detect print artifacts like banding or color shifts."""
        blemishes = []
        
        # Convert to LAB color space for better color analysis
        lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)
        
        # Detect banding in L channel
        grad_x = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(l_channel, cv2.CV_64F, 0, 1, ksize=3)
        
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Find regions with unnatural gradients
        _, artifact_mask = cv2.threshold(
            np.uint8(np.clip(gradient_magnitude, 0, 255)),
            int(30 + (1-self.sensitivity) * 30), 255, cv2.THRESH_BINARY
        )
        
        # Find contours
        contours, _ = cv2.findContours(artifact_mask, cv2.RETR_EXTERNAL, 
                                       cv2.CHAIN_APPROX_SIMPLE)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            if area < self.min_defect_size * 5:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            
            confidence = min(1.0, area / 1000) * self.sensitivity
            severity = self._get_severity(confidence)
            
            blemishes.append(Blemish(
                type=BlemishType.PRINT_ARTIFACT,
                confidence=confidence,
                bbox=(x, y, w, h),
                severity=severity
            ))
        
        return blemishes
    
    def _detect_border_damage(self, image: np.ndarray) -> List[Blemish]:
        """Detect damage to card borders."""
        blemishes = []
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        h, w = gray.shape
        border_width = int(min(h, w) * 0.05)  # 5% of smaller dimension
        
        # Check each border
        borders = {
            'top': gray[:border_width, :],
            'bottom': gray[-border_width:, :],
            'left': gray[:, :border_width],
            'right': gray[:, -border_width:]
        }
        
        for border_name, border_region in borders.items():
            # Look for irregularities in border
            edges = cv2.Canny(border_region, 50, 150)
            edge_density = np.sum(edges > 0) / edges.size
            
            if edge_density > 0.1 + (1 - self.sensitivity) * 0.2:
                # Map border name to coordinates
                if border_name == 'top':
                    bbox = (0, 0, w, border_width)
                elif border_name == 'bottom':
                    bbox = (0, h - border_width, w, border_width)
                elif border_name == 'left':
                    bbox = (0, 0, border_width, h)
                else:  # right
                    bbox = (w - border_width, 0, border_width, h)
                
                confidence = min(1.0, edge_density * 5) * self.sensitivity
                severity = self._get_severity(confidence)
                
                blemishes.append(Blemish(
                    type=BlemishType.BORDER_DAMAGE,
                    confidence=confidence,
                    bbox=bbox,
                    severity=severity
                ))
        
        return blemishes
    
    def _get_severity(self, confidence: float) -> str:
        """Determine severity based on confidence."""
        if confidence > 0.8:
            return "high"
        elif confidence > 0.5:
            return "medium"
        return "low"
    
    def create_inpaint_mask(self, image: np.ndarray, blemishes: List[Blemish]) -> np.ndarray:
        """Create a combined mask for inpainting all detected blemishes."""
        h, w = image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        
        for blemish in blemishes:
            x, y, bw, bh = blemish.bbox
            
            # Add padding around blemish
            padding = 5
            x1 = max(0, x - padding)
            y1 = max(0, y - padding)
            x2 = min(w, x + bw + padding)
            y2 = min(h, y + bh + padding)
            
            if blemish.mask is not None and blemish.mask.shape[0] > 0 and blemish.mask.shape[1] > 0:
                # Use provided mask
                mask_h, mask_w = y2 - y1, x2 - x1
                if blemish.mask.shape == (mask_h, mask_w):
                    mask[y1:y2, x1:x2] = cv2.bitwise_or(mask[y1:y2, x1:x2], blemish.mask)
                else:
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
            else:
                # Draw filled rectangle
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
        
        # Dilate mask slightly for better inpainting
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
        
        return mask


class BlemishRemover:
    """Remove detected blemishes using various inpainting techniques."""
    
    def __init__(self):
        self.inpaint_radius = 3
    
    def remove_blemishes(self, image: np.ndarray, blemishes: List[Blemish],
                        preserve_holographic: bool = True) -> np.ndarray:
        """Remove all detected blemishes from the image."""
        if not blemishes:
            return image
        
        # Create mask
        detector = BlemishDetector()
        mask = detector.create_inpaint_mask(image, blemishes)
        
        # Detect holographic regions if needed
        if preserve_holographic:
            holographic_mask = self._detect_holographic_regions(image)
            # Exclude holographic regions from inpainting
            mask = cv2.bitwise_and(mask, cv2.bitwise_not(holographic_mask))
        
        # Apply inpainting
        result = self._inpaint(image, mask)
        
        return result
    
    def _detect_holographic_regions(self, image: np.ndarray) -> np.ndarray:
        """Detect holographic/foil regions that should be preserved."""
        h, w = image.shape[:2]
        
        # Convert to HSV for better color analysis
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        
        # Holographic regions typically have:
        # - High saturation
        # - High value (brightness)
        # - Colorful appearance (high hue variation)
        
        saturation = hsv[:, :, 1]
        value = hsv[:, :, 2]
        
        # Detect high saturation and value regions
        _, sat_mask = cv2.threshold(saturation, 150, 255, cv2.THRESH_BINARY)
        _, val_mask = cv2.threshold(value, 200, 255, cv2.THRESH_BINARY)
        
        # Combine masks
        holographic_mask = cv2.bitwise_and(sat_mask, val_mask)
        
        # Detect high local variance (indicates iridescence)
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        local_var = cv2.Laplacian(gray, cv2.CV_64F)
        local_var = np.uint8(np.clip(np.abs(local_var), 0, 255))
        
        _, var_mask = cv2.threshold(local_var, 50, 255, cv2.THRESH_BINARY)
        
        # Final holographic mask
        holographic_mask = cv2.bitwise_and(holographic_mask, var_mask)
        
        # Dilate to cover full holographic regions
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        holographic_mask = cv2.dilate(holographic_mask, kernel, iterations=2)
        
        return holographic_mask
    
    def _inpaint(self, image: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Apply inpainting to remove blemishes."""
        # Use Telea inpainting method (good for small defects)
        result = cv2.inpaint(image, mask, self.inpaint_radius, cv2.INPAINT_TELEA)
        
        # For larger regions, also try NS method and blend
        if np.sum(mask) > 1000:  # If mask is large
            result_ns = cv2.inpaint(image, mask, self.inpaint_radius * 2, cv2.INPAINT_NS)
            # Blend results
            result = cv2.addWeighted(result, 0.6, result_ns, 0.4, 0)
        
        return result
    
    def remove_scratches_deep(self, image: np.ndarray, 
                              model: Optional[torch.nn.Module] = None) -> np.ndarray:
        """Remove scratches using deep learning inpainting (if model available)."""
        if model is None:
            # Fall back to traditional method
            detector = BlemishDetector(sensitivity=0.8)
            blemishes = detector._detect_scratches(image)
            return self.remove_blemishes(image, blemishes)
        
        # Deep learning approach would go here
        # This is a placeholder for integration with models like LaMa
        return image
