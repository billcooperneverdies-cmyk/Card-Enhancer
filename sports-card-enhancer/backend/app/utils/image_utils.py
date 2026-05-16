"""Image processing utilities."""
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ExifTags
from typing import Tuple, Optional, List
import io
import os


class ImageProcessor:
    """Utility class for image processing operations."""
    
    @staticmethod
    def load_image(path: str) -> np.ndarray:
        """Load image from path and return as numpy array."""
        # Try PIL first for better format support
        try:
            pil_img = Image.open(path)
            # Convert to RGB if necessary
            if pil_img.mode in ('RGBA', 'LA', 'P'):
                pil_img = pil_img.convert('RGB')
            return np.array(pil_img)
        except Exception:
            # Fallback to OpenCV
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Could not load image: {path}")
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    @staticmethod
    def save_image(image: np.ndarray, path: str, quality: int = 95, dpi: int = 300) -> None:
        """Save numpy array as image with format-specific options."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ext = os.path.splitext(path)[1].lower()
        
        # Convert to PIL for better format support and DPI handling
        pil_image = Image.fromarray(image)
        
        if ext in ['.jpg', '.jpeg']:
            # JPEG with quality and DPI
            pil_image.save(path, format='JPEG', quality=quality, dpi=(dpi, dpi))
        elif ext == '.png':
            # PNG with DPI (compression level 6 is a good balance)
            pil_image.save(path, format='PNG', compress_level=6, dpi=(dpi, dpi))
        elif ext == '.webp':
            # WebP with quality and method (6 = slowest but best compression)
            pil_image.save(path, format='WEBP', quality=quality, method=6)
        elif ext in ['.tiff', '.tif']:
            # TIFF with LZW compression and DPI
            pil_image.save(path, format='TIFF', compression='tiff_lzw', dpi=(dpi, dpi))
        else:
            # Default to PNG
            pil_image.save(path, format='PNG', dpi=(dpi, dpi))
    
    @staticmethod
    def resize_image(image: np.ndarray, max_dimension: int) -> np.ndarray:
        """Resize image while maintaining aspect ratio."""
        height, width = image.shape[:2]
        
        if max(height, width) <= max_dimension:
            return image
        
        scale = max_dimension / max(height, width)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        return cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
    
    @staticmethod
    def pad_to_multiple(image: np.ndarray, multiple: int = 8) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        """Pad image to make dimensions multiples of specified value."""
        h, w = image.shape[:2]
        
        # Calculate padding
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        
        # Distribute padding evenly
        pad_top = pad_h // 2
        pad_bottom = pad_h - pad_top
        pad_left = pad_w // 2
        pad_right = pad_w - pad_left
        
        # Pad image
        if len(image.shape) == 3:
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                           mode='reflect')
        else:
            padded = np.pad(image, ((pad_top, pad_bottom), (pad_left, pad_right)), 
                           mode='reflect')
        
        return padded, (pad_top, pad_bottom, pad_left, pad_right)
    
    @staticmethod
    def remove_padding(image: np.ndarray, pads: Tuple[int, int, int, int]) -> np.ndarray:
        """Remove padding from image."""
        pad_top, pad_bottom, pad_left, pad_right = pads
        h, w = image.shape[:2]
        
        if pad_bottom == 0:
            pad_bottom = None
        if pad_right == 0:
            pad_right = None
            
        return image[pad_top:h-pad_bottom if pad_bottom else h, 
                    pad_left:w-pad_right if pad_right else w]
    
    @staticmethod
    def auto_rotate(image: np.ndarray, pil_image: Optional[Image.Image] = None) -> np.ndarray:
        """Auto-rotate image based on EXIF orientation."""
        if pil_image is None:
            return image
            
        try:
            exif = pil_image._getexif()
            if exif is not None:
                orientation = exif.get(274)  # 274 is Orientation tag
                if orientation == 3:
                    return cv2.rotate(image, cv2.ROTATE_180)
                elif orientation == 6:
                    return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                elif orientation == 8:
                    return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        except Exception:
            pass
        
        return image
    
    @staticmethod
    def detect_card_border(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        """Detect card border for automatic cropping."""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Find largest contour (assumed to be the card)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Filter out small detections
        img_area = image.shape[0] * image.shape[1]
        contour_area = w * h
        
        if contour_area < img_area * 0.1:  # Less than 10% of image
            return None
        
        return (x, y, w, h)
    
    @staticmethod
    def crop_to_aspect_ratio(image: np.ndarray, target_ratio: float = 2.5/3.5) -> np.ndarray:
        """Crop image to target aspect ratio (default: standard card ratio)."""
        h, w = image.shape[:2]
        current_ratio = w / h
        
        if abs(current_ratio - target_ratio) < 0.01:
            return image
        
        if current_ratio > target_ratio:
            # Image is too wide, crop width
            new_w = int(h * target_ratio)
            start_x = (w - new_w) // 2
            return image[:, start_x:start_x + new_w]
        else:
            # Image is too tall, crop height
            new_h = int(w / target_ratio)
            start_y = (h - new_h) // 2
            return image[start_y:start_y + new_h, :]


class ImageEnhancer:
    """Image enhancement operations."""
    
    @staticmethod
    def sharpen(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """Apply sharpening filter."""
        pil_img = Image.fromarray(image)
        
        # Create sharpening kernel
        factor = 1 + amount
        kernel = np.array([[-1, -1, -1],
                          [-1,  factor + 8, -1],
                          [-1, -1, -1]]) * (amount / 8)
        kernel[1, 1] = factor
        
        sharpened = pil_img.filter(ImageFilter.Kernel((3, 3), kernel.flatten(), scale=factor))
        return np.array(sharpened)
    
    @staticmethod
    def adjust_color_temperature(image: np.ndarray, temperature: float) -> np.ndarray:
        """Adjust color temperature. Negative = cooler, Positive = warmer."""
        pil_img = Image.fromarray(image)
        r, g, b = pil_img.split()
        
        # Adjust red and blue channels
        r = r.point(lambda i: min(255, int(i * (1 + temperature * 0.1))))
        b = b.point(lambda i: min(255, int(i * (1 - temperature * 0.1))))
        
        return np.array(Image.merge('RGB', (r, g, b)))
    
    @staticmethod
    def adjust_saturation(image: np.ndarray, factor: float) -> np.ndarray:
        """Adjust image saturation."""
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Color(pil_img)
        return np.array(enhancer.enhance(factor))
    
    @staticmethod
    def adjust_contrast(image: np.ndarray, amount: float) -> np.ndarray:
        """Adjust image contrast."""
        pil_img = Image.fromarray(image)
        enhancer = ImageEnhance.Contrast(pil_img)
        # Map 0-1 to 0.5-1.5 range
        factor = 0.5 + amount
        return np.array(enhancer.enhance(factor))
    
    @staticmethod
    def reduce_noise(image: np.ndarray, strength: float = 0.5) -> np.ndarray:
        """Apply noise reduction."""
        # Map strength to filter size
        h = int(3 + strength * 7)  # 3 to 10
        h = h if h % 2 == 1 else h + 1  # Ensure odd
        
        return cv2.fastNlMeansDenoisingColored(image, None, h, h, 7, 21)
    
    @staticmethod
    def enhance_details(image: np.ndarray, amount: float = 0.5) -> np.ndarray:
        """Enhance fine details using unsharp mask."""
        pil_img = Image.fromarray(image)
        
        # Unsharp mask parameters based on amount
        radius = 2
        percent = int(50 + amount * 150)  # 50% to 200%
        threshold = 3
        
        enhanced = pil_img.filter(
            ImageFilter.UnsharpMask(radius=radius, percent=percent, threshold=threshold)
        )
        
        return np.array(enhanced)
