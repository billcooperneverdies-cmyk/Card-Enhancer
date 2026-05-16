"""Shared image processing utilities to prevent code duplication."""
import cv2
import numpy as np
from typing import List, Optional, Tuple


def apply_clahe_and_detect_edges(
    image: np.ndarray, 
    clip_limit: float = 2.0, 
    tile_size: int = 8,
    canny_low: int = 50,
    canny_high: int = 150
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply CLAHE contrast enhancement and Canny edge detection.
    
    Args:
        image: Input RGB image
        clip_limit: CLAHE contrast limit
        tile_size: CLAHE grid size
        canny_low: Lower Canny threshold
        canny_high: Upper Canny threshold
        
    Returns:
        Tuple of (enhanced_gray, edges)
    """
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Enhance contrast
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
    enhanced = clahe.apply(gray)
    
    # Edge detection
    edges = cv2.Canny(enhanced, canny_low, canny_high)
    
    # Dilate to connect segments
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    return enhanced, dilated


def find_hough_lines(
    edge_image: np.ndarray, 
    sensitivity: float = 1.0
) -> Optional[np.ndarray]:
    """
    Detect lines using Hough Transform with sensitivity adjustment.
    
    Args:
        edge_image: Preprocessed edge image
        sensitivity: 0.0 (low) to 1.0 (high)
        
    Returns:
        Detected lines or None
    """
    threshold = int(20 + (1 - sensitivity) * 30)
    min_length = int(30 + (1 - sensitivity) * 50)
    
    return cv2.HoughLinesP(
        edge_image, 
        1, 
        np.pi/180,
        threshold=threshold,
        minLineLength=min_length,
        maxLineGap=10
    )


def filter_defect_contours(
    contours: List[np.ndarray], 
    min_size: int = 10, 
    max_size: int = 500,
    circularity_threshold: float = 0.5
) -> List[Tuple[np.ndarray, float]]:
    """
    Filter contours based on area and circularity to identify defects.
    
    Args:
        contours: List of detected contours
        min_size: Minimum area threshold
        max_size: Maximum area threshold
        circularity_threshold: Minimum circularity score
        
    Returns:
        List of tuples (contour, circularity_score)
    """
    valid_defects = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_size or area > max_size:
            continue

        perimeter = cv2.arcLength(contour, True)
        if perimeter > 0:
            circularity = 4 * np.pi * area / (perimeter ** 2)
            if circularity > circularity_threshold:
                valid_defects.append((contour, circularity))
                
    return valid_defects


def denoise_image(image: np.ndarray, h: int = 10) -> np.ndarray:
    """Apply non-local means denoising to colored images."""
    return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
