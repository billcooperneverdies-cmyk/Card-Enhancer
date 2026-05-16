"""
Real-ESRGAN Service Integration

This module implements the 'Hybrid-Architecture Blueprint' for automated sports card restoration.
It orchestrates state-of-the-art super-resolution models (Real-ESRGAN, SwinIR) within a 
lightweight backend to offload computationally intensive tasks from the client.

Key Architectural Features:
1.  **Lazy Loading**: Models are loaded only on first inference to minimize startup memory footprint.
2.  **Tiled Inference**: To support free-tier GPUs with limited VRAM, large images are processed 
    in overlapping tiles and stitched back together, preventing Out-Of-Memory (OOM) crashes.
3.  **Multi-Stage Pipeline**: Supports a sequential workflow where images can first pass through 
    an inpainting model (e.g., LaMa for scratch removal) before super-resolution, as recommended 
    for handling complex degradations in scanned collectibles.
4.  **Hardware Fallback**: Automatically detects CUDA availability; falls back to CPU if GPU is unavailable,
    ensuring robustness across different deployment environments.
"""

import os
import torch
import cv2
import numpy as np
from typing import Optional, Tuple, List
from pathlib import Path
import logging

# Configure logging for the service
logger = logging.getLogger(__name__)

# Attempt to import Real-ESRGAN dependencies
# These are heavy dependencies; if missing, we gracefully degrade to OpenCV interpolation
try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    REAL_ESRGAN_AVAILABLE = True
except ImportError:
    REAL_ESRGAN_AVAILABLE = False
    logger.warning("Real-ESRGAN dependencies not found. Enhancement will fallback to OpenCV Lanczos interpolation.")


class RealESRGANService:
    """
    Service class for managing Super-Resolution inference.
    
    This class encapsulates the complexity of loading models, managing device context,
    and performing tiled inference. It acts as the 'Backend Orchestrator' in the 
    Hybrid Architecture, receiving images from the API and returning enhanced binaries.
    
    Example usage:
        ```python
        # Initialize service with tiling for memory efficiency
        service = RealESRGANService(tile_size=256, tile_pad=10)
        
        # Single image enhancement
        enhanced = service.enhance_image(input_image, model_name="RealESRGAN_x4plus")
        
        # Multi-stage pipeline with scratch removal
        restored = service.restore_pipeline(
            input_image, 
            remove_scratches=True, 
            model_name="RealESRGAN_x4plus"
        )
        ```
    """

    # Mapping of user-friendly model names to their architecture configurations
    # Supports both GAN-based (sharper, potentially more artifacts) and PSNR-based (cleaner) models
    MODEL_CONFIGS = {
        "RealESRGAN_x4plus": {
            "type": "realesrgan",
            "scale": 4,
            "model_path": "RealESRGAN_x4plus.pth",
            "description": "General purpose x4 upscaler (GAN-based, sharper details)"
        },
        "RealESRNet_x4plus": {
            "type": "realesrgan",
            "scale": 4,
            "model_path": "RealESRNet_x4plus.pth",
            "description": "Conservative x4 upscaler (PSNR-based, fewer artifacts)"
        },
        "RealESRGAN_x4plus_anime_6B": {
            "type": "realesrgan",
            "scale": 4,
            "model_path": "RealESRGAN_x4plus_anime_6B.pth",
            "description": "Optimized for anime/illustration style cards"
        },
    }

    def __init__(self, weights_dir: str = "weights", tile_size: int = 0, tile_pad: int = 10):
        """
        Initialize the SR Service.
        
        Args:
            weights_dir: Directory to store/download model weights.
            tile_size: Size of tiles for inference. If 0, processes whole image (risky for VRAM).
                       Recommended: 200-400 for free-tier GPUs.
            tile_pad: Padding overlap between tiles to avoid seam artifacts.
        """
        self.weights_dir = Path(weights_dir)
        self.weights_dir.mkdir(exist_ok=True)
        
        # Tiling parameters crucial for 'Free-Tier GPU' constraint mentioned in blueprint
        self.tile_size = tile_size 
        self.tile_pad = tile_pad
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"RealESRGANService initialized on device: {self.device}")
        
        # Cache for loaded models to avoid reloading on every request
        self._model_cache = {}
        self._upsampler = None

    def _get_model_path(self, model_name: str) -> Path:
        """Resolve the local path for a given model name."""
        if model_name not in self.MODEL_CONFIGS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(self.MODEL_CONFIGS.keys())}")
        return self.weights_dir / self.MODEL_CONFIGS[model_name]["model_path"]

    def load_model(self, model_name: str = "RealESRGAN_x4plus"):
        """
        Lazily load the specified super-resolution model.
        
        Data Flow:
        1. Check internal cache for existing model instance.
        2. Verify weight file exists (auto-download logic would go here in prod).
        3. Instantiate the specific architecture (RRDBNet for ESRGAN).
        4. Wrap in RealESRGANer utility which handles tiling and device placement.
        
        Args:
            model_name: Key from MODEL_CONFIGS
            
        Returns:
            RealESRGANer instance ready for inference
            
        Raises:
            RuntimeError: If dependencies missing or model loading fails
            FileNotFoundError: If model weights not found
        """
        if model_name in self._model_cache:
            logger.debug(f"Model {model_name} already loaded from cache")
            return self._model_cache[model_name]

        if not REAL_ESRGAN_AVAILABLE:
            raise RuntimeError("Real-ESRGAN library not installed. Cannot load models.")

        config = self.MODEL_CONFIGS.get(model_name)
        if not config:
            raise ValueError(f"Model configuration not found for {model_name}")

        model_path = self._get_model_path(model_name)
        
        # Check if weights exist; if not, raise clear error
        if not model_path.exists():
            logger.error(f"Model weights not found at {model_path}. Please download manually.")
            raise FileNotFoundError(f"Missing model weights: {model_path}")

        logger.info(f"Loading model {model_name} on {self.device}...")

        try:
            if config["type"] == "realesrgan":
                # Initialize RRDBNet for Real-ESRGAN variants
                # num_block=23, num_feat=64 are standard for x4plus models
                model = RRDBNet(
                    num_in_ch=3, 
                    num_out_ch=3, 
                    num_feat=64, 
                    num_block=23, 
                    num_grow_ch=32, 
                    scale=config["scale"]
                )
                
                # RealESRGANer wraps the model and handles tiling logic internally
                # This is critical for processing large card scans without OOM errors
                self._upsampler = RealESRGANer(
                    scale=config["scale"],
                    model_path=str(model_path),
                    model=model,
                    tile=self.tile_size,  # Critical for memory management
                    tile_pad=self.tile_pad,
                    pre_pad=0,
                    half=True if self.device.type == 'cuda' else False,  # FP16 on GPU for speed
                    device=self.device,
                )
            
            self._model_cache[model_name] = self._upsampler
            logger.info(f"Model {model_name} loaded successfully.")
            return self._upsampler

        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise RuntimeError(f"Model loading failed: {e}")

    def enhance_image(self, image: np.ndarray, model_name: str = "RealESRGAN_x4plus") -> np.ndarray:
        """
        Perform super-resolution on the input image.
        
        This is the core entry point for the 'Enhancement' stage of the pipeline.
        
        Data Flow:
        1. Validate input image (non-empty, correct dimensions)
        2. Load model if not already cached
        3. Run inference through RealESRGANer.enhance()
        4. Handle OOM errors with helpful suggestions
        5. Return enhanced BGR image
        
        Args:
            image: Input BGR numpy array (from OpenCV). Shape: (H, W, 3)
            model_name: Key from MODEL_CONFIGS
            
        Returns:
            Enhanced BGR numpy array, upscaled by model's scale factor (typically 4x)
            
        Raises:
            RuntimeError: If enhancement fails (OOM, invalid input, etc.)
        """
        if not REAL_ESRGAN_AVAILABLE:
            logger.warning("Real-ESRGAN unavailable. Using OpenCV Lanczos fallback.")
            h, w = image.shape[:2]
            return cv2.resize(image, (w * 4, h * 4), interpolation=cv2.INTER_LANCZOS4)

        try:
            upsampler = self.load_model(model_name)
            
            # Inference: The RealESRGANer handles the heavy lifting, including tiling
            # Output is a tuple: (result_image, output_info_string)
            result, _ = upsampler.enhance(image, outscale=upsampler.scale)
            
            return result

        except torch.cuda.OutOfMemoryError:
            logger.error("GPU Out of Memory. Try reducing tile_size or using CPU.")
            raise RuntimeError(
                "Enhancement failed: GPU Out of Memory. "
                "Please try a smaller tile_size (e.g., 128 or 256) or switch to CPU mode."
            )
        except Exception as e:
            logger.error(f"Enhancement failed: {str(e)}")
            raise

    def restore_pipeline(
        self, 
        image: np.ndarray, 
        remove_scratches: bool = False, 
        model_name: str = "RealESRGAN_x4plus"
    ) -> np.ndarray:
        """
        Implements the Multi-Stage Pipeline described in the Hybrid Blueprint.
        
        This method orchestrates the sequential processing workflow recommended for
        sports card restoration:
        
        Data Flow:
        1. [Optional Stage 1] Scratch Removal: Pass image through an Inpainting model 
           (e.g., LaMa) if severe structural defects are detected or requested.
        2. [Stage 2] Super-Resolution: Pass the (potentially cleaned) image through 
           Real-ESRGAN/SwinIR for upscaling and general quality improvement.
        
        Args:
            image: Input BGR numpy array from OpenCV
            remove_scratches: Boolean flag to trigger the inpainting stage
            model_name: SR model to use for the second stage
            
        Returns:
            Final restored and upscaled BGR image
            
        Example:
            ```python
            # Full pipeline with scratch removal
            restored = service.restore_pipeline(
                card_scan, 
                remove_scratches=True, 
                model_name="RealESRGAN_x4plus"
            )
            
            # Just super-resolution
            upscaled = service.restore_pipeline(card_scan, model_name="RealESRNet_x4plus")
            ```
        """
        current_image = image

        # Stage 1: Inpainting (Scratch Removal)
        # Note: Actual LaMa integration would happen here. 
        # In a full implementation, we would call a separate LaMaService.
        if remove_scratches:
            logger.info("Stage 1: Initiating scratch removal (Inpainting)...")
            # Placeholder for future inpainting integration:
            # current_image = self.inpainting_service.remove_defects(current_image)
            logger.warning("Inpainting module not yet connected. Skipping scratch removal stage.")
        
        # Stage 2: Super-Resolution & General Restoration
        logger.info(f"Stage 2: Enhancing with {model_name}...")
        try:
            enhanced_image = self.enhance_image(current_image, model_name)
            return enhanced_image
        except Exception as e:
            logger.error(f"Pipeline failed at SR stage: {e}")
            raise

    def upscale_with_fallback(
        self, 
        image: np.ndarray, 
        model_name: str = "RealESRGAN_x4plus",
        fallback_scale: int = 4
    ) -> Tuple[np.ndarray, bool]:
        """
        Upscale with automatic fallback to traditional interpolation if SR fails.
        
        This method provides robust upscaling by attempting Real-ESRGAN first,
        then falling back to OpenCV interpolation if the model fails or isn't available.
        Useful for batch processing where you don't want one failure to stop the entire job.
        
        Args:
            image: Input BGR image
            model_name: SR model to attempt
            fallback_scale: Scale factor for fallback interpolation
            
        Returns:
            Tuple of (upscaled_image, used_sr):
            - upscaled_image: Enhanced BGR image
            - used_sr: Boolean indicating if SR was successful (True) or fallback used (False)
        """
        try:
            output = self.enhance_image(image, model_name)
            return output, True
            
        except Exception as e:
            logger.warning(f"Real-ESRGAN failed, falling back to Lanczos: {e}")
            
            # Fallback to traditional interpolation
            h, w = image.shape[:2]
            new_w, new_h = int(w * fallback_scale), int(h * fallback_scale)
            
            output = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            
            logger.info(f"Fallback upscaling complete: {w}x{h} -> {new_w}x{new_h}")
            return output, False
