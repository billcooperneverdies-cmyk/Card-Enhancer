"""Main enhancement service that orchestrates all image processing."""
import cv2
import numpy as np
from typing import Optional, Tuple
from PIL import Image
import logging

from app.utils.image_utils import ImageProcessor, ImageEnhancer
from app.services.blemish_detector import BlemishDetector, BlemishRemover
from app.models.schemas import EnhancementSettings, BlemishDetection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EnhancementService:
    """Main service for enhancing sports card images."""
    
    def __init__(self):
        self.image_processor = ImageProcessor()
        self.image_enhancer = ImageEnhancer()
        self.blemish_detector = BlemishDetector()
        self.blemish_remover = BlemishRemover()
        self.upscaler = None  # Lazy load
        
    def enhance(self, image_path: str, settings: EnhancementSettings) -> Tuple[np.ndarray, list]:
        """
        Enhance a single card image.
        
        Returns:
            Tuple of (enhanced_image, detected_blemishes)
        """
        logger.info(f"Starting enhancement for {image_path}")
        
        # Load image
        image = self.image_processor.load_image(image_path)
        original_shape = image.shape
        
        # Resize if too large
        image = self.image_processor.resize_image(image, 4096)
        
        detected_blemishes = []
        
        # Step 1: Blemish Detection and Removal
        if settings.blemish_removal:
            logger.info("Detecting blemishes...")
            blemishes = self.blemish_detector.detect_all(image)
            detected_blemishes = self._convert_blemishes(blemishes)
            
            if blemishes:
                logger.info(f"Found {len(blemishes)} blemishes, removing...")
                image = self.blemish_remover.remove_blemishes(
                    image, blemishes, 
                    preserve_holographic=settings.preserve_holographic
                )
        
        # Step 2: Noise Reduction (do early to prevent amplifying noise)
        if settings.noise_reduction:
            logger.info("Applying noise reduction...")
            image = self.image_enhancer.reduce_noise(
                image, 
                strength=settings.noise_reduction_strength
            )
        
        # Step 3: Color Correction
        if settings.color_correction:
            logger.info("Applying color correction...")
            image = self.image_enhancer.adjust_color_temperature(
                image, 
                temperature=settings.color_temperature
            )
            image = self.image_enhancer.adjust_saturation(
                image, 
                factor=settings.saturation
            )
        
        # Step 4: Contrast Enhancement
        if settings.contrast_enhancement:
            logger.info("Enhancing contrast...")
            image = self.image_enhancer.adjust_contrast(
                image, 
                amount=settings.contrast_amount
            )
        
        # Step 5: Sharpening
        if settings.sharpening:
            logger.info("Applying sharpening...")
            image = self.image_enhancer.sharpen(
                image, 
                amount=settings.sharpening_amount
            )
            image = self.image_enhancer.enhance_details(
                image, 
                amount=settings.sharpening_amount
            )
        
        # Step 6: Upscaling (if requested)
        if settings.upscaling and settings.upscale_factor > 1:
            logger.info(f"Upscaling by factor of {settings.upscale_factor}...")
            image = self._upscale(image, settings.upscale_factor)
        
        # Resize back to original if needed
        if image.shape != original_shape:
            image = cv2.resize(image, (original_shape[1], original_shape[0]), 
                             interpolation=cv2.INTER_LANCZOS4)
        
        logger.info("Enhancement complete")
        return image, detected_blemishes
    
    def generate_preview(self, image_path: str, settings: EnhancementSettings,
                        max_size: int = 1024) -> Tuple[np.ndarray, list]:
        """Generate a preview of the enhancement (faster, lower quality)."""
        # Load and resize for preview
        image = self.image_processor.load_image(image_path)
        image = self.image_processor.resize_image(image, max_size)
        
        # Use faster settings for preview
        preview_settings = EnhancementSettings(
            blemish_removal=settings.blemish_removal,
            blemish_sensitivity=settings.blemish_sensitivity,
            sharpening=settings.sharpening,
            sharpening_amount=settings.sharpening_amount * 0.7,
            color_correction=settings.color_correction,
            color_temperature=settings.color_temperature,
            saturation=settings.saturation,
            contrast_enhancement=settings.contrast_enhancement,
            contrast_amount=settings.contrast_amount,
            noise_reduction=False,  # Skip for preview
            upscaling=False,  # Skip for preview
            preserve_holographic=settings.preserve_holographic,
            output_format=settings.output_format,
            output_quality=85  # Lower quality for preview
        )
        
        # Enhance with preview settings
        enhanced, blemishes = self.enhance(image_path, preview_settings)
        
        # Resize to preview size
        enhanced = self.image_processor.resize_image(enhanced, max_size)
        
        return enhanced, blemishes
    
    def _upscale(self, image: np.ndarray, factor: int) -> np.ndarray:
        """Upscale image using super-resolution."""
        # Try to use Real-ESRGAN if available
        if self.upscaler is None:
            try:
                from realesrgan import RealESRGANer
                from basicsr.archs.rrdbnet_arch import RRDBNet
                
                model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,
                               num_block=23, num_grow_ch=32, scale=4)
                
                self.upscaler = RealESRGANer(
                    scale=4,
                    model_path='https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth',
                    model=model,
                    tile=0,
                    pre_pad=0,
                    half=True
                )
            except Exception as e:
                logger.warning(f"Could not load Real-ESRGAN: {e}")
                self.upscaler = None
        
        if self.upscaler:
            try:
                output, _ = self.upscaler.enhance(image, outscale=factor)
                return output
            except Exception as e:
                logger.warning(f"Real-ESRGAN enhancement failed: {e}")
        
        # Fallback to Lanczos interpolation
        h, w = image.shape[:2]
        new_h, new_w = h * factor, w * factor
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    def _convert_blemishes(self, blemishes: list) -> list:
        """Convert internal blemish objects to API schema."""
        result = []
        for b in blemishes:
            result.append(BlemishDetection(
                type=b.type.value,
                confidence=b.confidence,
                bbox=list(b.bbox),
                severity=b.severity
            ))
        return result
    
    def auto_crop_card(self, image_path: str) -> Optional[np.ndarray]:
        """Automatically detect and crop card from image."""
        image = self.image_processor.load_image(image_path)
        
        # Try to detect card border
        border = self.image_processor.detect_card_border(image)
        
        if border:
            x, y, w, h = border
            cropped = image[y:y+h, x:x+w]
            
            # Crop to standard card aspect ratio
            cropped = self.image_processor.crop_to_aspect_ratio(cropped)
            
            return cropped
        
        # Fallback to aspect ratio crop
        return self.image_processor.crop_to_aspect_ratio(image)


class BatchEnhancementService:
    """Service for batch processing multiple images."""
    
    def __init__(self):
        self.enhancement_service = EnhancementService()
    
    async def process_batch(self, image_paths: list, settings: EnhancementSettings,
                           progress_callback=None) -> list:
        """Process multiple images with progress updates."""
        results = []
        total = len(image_paths)
        
        for i, path in enumerate(image_paths):
            try:
                if progress_callback:
                    await progress_callback(i, total, f"Processing image {i+1}/{total}")
                
                enhanced, blemishes = self.enhancement_service.enhance(path, settings)
                results.append({
                    'path': path,
                    'success': True,
                    'enhanced': enhanced,
                    'blemishes': blemishes
                })
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                results.append({
                    'path': path,
                    'success': False,
                    'error': str(e)
                })
        
        if progress_callback:
            await progress_callback(total, total, "Batch complete")
        
        return results
