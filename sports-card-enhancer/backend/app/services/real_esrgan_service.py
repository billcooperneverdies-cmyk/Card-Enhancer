"""
Real-ESRGAN Super-Resolution Service for Card Enhancement.

This module provides a clean integration layer between the Card-Enhancer project
and the Real-ESRGAN super-resolution models. It handles model loading, inference,
and provides fallback mechanisms for when GPU/CUDA is not available.

Data Flow:
    1. Input image (numpy array in BGR format from OpenCV) 
       -> RealESRGANService.upscale()
    2. Image is converted to RGB and normalized to [0, 1] range
    3. Optional tiling for large images to prevent OOM errors
    4. Model inference on GPU (with CPU fallback)
    5. Output image returned as numpy array in BGR format (OpenCV compatible)
"""
import os
import logging
from typing import Optional, Tuple, Literal
from enum import Enum
import numpy as np
import torch
import cv2

logger = logging.getLogger(__name__)


class SRModelType(str, Enum):
    """
    Supported Super-Resolution model types.
    
    Each model has different characteristics:
    - RealESRGAN_x4plus: Best general purpose model, good for photos
    - RealESRNet_x4plus: More conservative, less artifacting
    - RealESRGAN_x4plus_anime_6B: Optimized for anime/illustrations
    - RealESRGAN_x2plus: 2x upscaling, faster than 4x
    - realesr-animevideov3: Lightweight model for video/anime
    - realesr-general-x4v3: General purpose with denoise control
    """
    REAL_ESRGAN_X4PLUS = "RealESRGAN_x4plus"
    REAL_ESRNET_X4PLUS = "RealESRNet_x4plus"
    REAL_ESRGAN_X4PLUS_ANIME = "RealESRGAN_x4plus_anime_6B"
    REAL_ESRGAN_X2PLUS = "RealESRGAN_x2plus"
    ANIME_VIDEO_V3 = "realesr-animevideov3"
    GENERAL_X4V3 = "realesr-general-x4v3"


# Model configuration mapping
# Maps model names to their architecture class, scale factor, and download URL
MODEL_CONFIGS = {
    SRModelType.REAL_ESRGAN_X4PLUS: {
        "arch": "RRDBNet",
        "scale": 4,
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
    },
    SRModelType.REAL_ESRNET_X4PLUS: {
        "arch": "RRDBNet",
        "scale": 4,
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth"
    },
    SRModelType.REAL_ESRGAN_X4PLUS_ANIME: {
        "arch": "RRDBNet",
        "scale": 4,
        "num_block": 6,  # Fewer blocks for anime style
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth"
    },
    SRModelType.REAL_ESRGAN_X2PLUS: {
        "arch": "RRDBNet",
        "scale": 2,
        "num_block": 23,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
    },
    SRModelType.ANIME_VIDEO_V3: {
        "arch": "SRVGGNetCompact",
        "scale": 4,
        "num_conv": 16,  # Lightweight
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth"
    },
    SRModelType.GENERAL_X4V3: {
        "arch": "SRVGGNetCompact",
        "scale": 4,
        "num_conv": 32,
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth"
    }
}


class RealESRGANService:
    """
    Service class for Real-ESRGAN super-resolution upscaling.
    
    This class provides a high-level interface for upscaling images using
    Real-ESRGAN models. It handles:
    - Lazy model loading (models are loaded only when first needed)
    - Automatic device selection (GPU with CUDA, or CPU fallback)
    - Tiled inference for large images to prevent OOM errors
    - Multiple model architectures for different use cases
    
    Example usage:
        ```python
        service = RealESRGANService(model_type=SRModelType.REAL_ESRGAN_X4PLUS)
        
        # Upscale an image
        enhanced_image = service.upscale(
            image=input_image,
            outscale=4.0,
            tile_size=256  # Use tiling for large images
        )
        
        # Batch upscale with progress callback
        for i, img in enumerate(images):
            result = service.upscale(img, outscale=2.0)
            progress_callback(i, len(images))
        ```
    """
    
    def __init__(
        self,
        model_type: SRModelType = SRModelType.REAL_ESRGAN_X4PLUS,
        model_path: Optional[str] = None,
        tile_size: int = 0,
        tile_pad: int = 10,
        pre_pad: int = 0,
        use_half_precision: bool = True,
        gpu_id: Optional[int] = None,
        dni_weight: Optional[float] = None
    ):
        """
        Initialize the Real-ESRGAN service.
        
        Args:
            model_type: The type of SR model to use (see SRModelType enum)
            model_path: Custom path to model weights. If None, will auto-download
            tile_size: Size of tiles for tiled inference. 0 = no tiling (process whole image)
                      Use 256-512 for GPUs with limited memory
            tile_pad: Padding size for each tile to prevent border artifacts
            pre_pad: Pre-padding size at image borders
            use_half_precision: Use FP16 for faster inference on supported GPUs
            gpu_id: Specific GPU device ID. None = auto-select first available
            dni_weight: Deep Network Interpolation weight (0-1) for blending models.
                       Only used with 'realesr-general-x4v3' model
        """
        self.model_type = model_type
        self.custom_model_path = model_path
        self.tile_size = tile_size
        self.tile_pad = tile_pad
        self.pre_pad = pre_pad
        self.use_half_precision = use_half_precision
        self.gpu_id = gpu_id
        self.dni_weight = dni_weight
        
        # Lazy-loaded components
        self._model = None
        self._upsampler = None
        self._device = None
        
        # Validate model type
        if model_type not in MODEL_CONFIGS:
            raise ValueError(f"Unsupported model type: {model_type}. "
                           f"Available: {list(SRModelType)}")
        
        logger.info(f"RealESRGANService initialized with model: {model_type.value}")
        logger.info(f"Device preference: GPU ID {gpu_id if gpu_id is not None else 'auto'}")
        logger.info(f"Tiling: {'enabled (' + str(tile_size) + 'px)' if tile_size > 0 else 'disabled'}")
    
    @property
    def device(self) -> torch.device:
        """Get the compute device (GPU or CPU), initializing if needed."""
        if self._device is None:
            self._device = self._select_device()
        return self._device
    
    @property
    def upsampler(self):
        """
        Get the RealESRGANer upsampler instance, loading the model if needed.
        
        This is a lazy-loading property - the model is only loaded when first accessed.
        """
        if self._upsampler is None:
            self._load_model()
        return self._upsampler
    
    def _select_device(self) -> torch.device:
        """
        Select the best available compute device.
        
        Priority:
        1. Specified GPU ID (if provided and available)
        2. First available CUDA GPU
        3. CPU fallback
        
        Returns:
            torch.device: The selected compute device
        """
        if self.gpu_id is not None:
            if torch.cuda.is_available() and self.gpu_id < torch.cuda.device_count():
                logger.info(f"Using specified GPU: cuda:{self.gpu_id}")
                return torch.device(f'cuda:{self.gpu_id}')
            else:
                logger.warning(f"Specified GPU:{self.gpu_id} not available, falling back to auto-select")
        
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            logger.info(f"CUDA available: {gpu_name} ({gpu_memory:.1f} GB)")
            return torch.device('cuda')
        else:
            logger.warning("CUDA not available - using CPU (inference will be slower)")
            return torch.device('cpu')
    
    def _load_model(self):
        """
        Load the Real-ESRGAN model and create the upsampler instance.
        
        This method:
        1. Imports required Real-ESRGAN modules
        2. Instantiates the correct architecture based on model_type
        3. Downloads pretrained weights if not cached
        4. Creates the RealESRGANer wrapper for inference
        
        Raises:
            ImportError: If Real-ESRGAN dependencies are not installed
            RuntimeError: If model loading fails
        """
        try:
            # Import Real-ESRGAN modules
            # These are imported here to avoid requiring the dependency unless actually used
            from realesrgan import RealESRGANer
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from basicsr.utils.download_util import load_file_from_url
            
            logger.info(f"Loading {self.model_type.value} model...")
            
            config = MODEL_CONFIGS[self.model_type]
            
            # Instantiate the model architecture
            if config["arch"] == "RRDBNet":
                model = RRDBNet(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_block=config["num_block"],
                    num_grow_ch=32,
                    scale=config["scale"]
                )
                logger.debug(f"Created RRDBNet with {config['num_block']} blocks, scale={config['scale']}")
                
            elif config["arch"] == "SRVGGNetCompact":
                act_type = 'prelu'  # Default activation for VGG-style models
                model = SRVGGNetCompact(
                    num_in_ch=3,
                    num_out_ch=3,
                    num_feat=64,
                    num_conv=config["num_conv"],
                    upscale=config["scale"],
                    act_type=act_type
                )
                logger.debug(f"Created SRVGGNetCompact with {config['num_conv']} conv layers, scale={config['scale']}")
            else:
                raise ValueError(f"Unknown architecture: {config['arch']}")
            
            # Determine model path (download if necessary)
            if self.custom_model_path:
                model_path = self.custom_model_path
                logger.info(f"Using custom model path: {model_path}")
            else:
                # Auto-download from GitHub releases
                model_path = load_file_from_url(
                    url=config["url"],
                    model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'weights'),
                    progress=True,
                    file_name=f"{self.model_type.value}.pth"
                )
                logger.info(f"Model downloaded/cached at: {model_path}")
            
            # Handle DNI (Deep Network Interpolation) for realesr-general-x4v3
            dni_weight_list = None
            if (self.model_type == SRModelType.GENERAL_X4V3 and 
                self.dni_weight is not None and 
                self.dni_weight != 1.0):
                # Load both WDN and non-WDN variants for interpolation
                wdn_url = config["url"].replace('realesr-general-x4v3', 'realesr-general-wdn-x4v3')
                wdn_path = load_file_from_url(
                    url=wdn_url,
                    model_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'weights'),
                    progress=True,
                    file_name=f"{self.model_type.value}_wdn.pth"
                )
                model_path = [model_path, wdn_path]
                dni_weight_list = [self.dni_weight, 1 - self.dni_weight]
                logger.info(f"DNI enabled: {self.dni_weight:.2f} (non-WDN) + {1-self.dni_weight:.2f} (WDN)")
            
            # Create the upsampler wrapper
            self._upsampler = RealESRGANer(
                scale=config["scale"],
                model_path=model_path,
                dni_weight=dni_weight_list,
                model=model,
                tile=self.tile_size,
                tile_pad=self.tile_pad,
                pre_pad=self.pre_pad,
                half=self.use_half_precision and self.device.type == 'cuda',
                device=self.device,
                gpu_id=self.gpu_id
            )
            
            logger.info(f"Model loaded successfully on {self.device}")
            
        except ImportError as e:
            logger.error(f"Failed to import Real-ESRGAN dependencies: {e}")
            logger.error("Please install: pip install realesrgan basicsr gfpgan")
            raise ImportError(
                "Real-ESRGAN not installed. Run: pip install realesrgan basicsr gfpgan"
            ) from e
        except Exception as e:
            logger.error(f"Failed to load Real-ESRGAN model: {e}")
            raise RuntimeError(f"Model loading failed: {e}") from e
    
    def upscale(
        self,
        image: np.ndarray,
        outscale: float = 4.0,
        alpha_upsampler: Literal['realesrgan', 'bicubic'] = 'realesrgan'
    ) -> Tuple[np.ndarray, str]:
        """
        Upscale an image using the loaded Real-ESRGAN model.
        
        This is the main entry point for super-resolution. It handles:
        - Input validation
        - Color space conversion (BGR <-> RGB)
        - Alpha channel processing (for PNG with transparency)
        - Model inference
        - Output formatting
        
        Args:
            image: Input image as numpy array (H x W x C) in BGR format (OpenCV standard)
                  Supports grayscale (H x W), RGB/BGR (H x W x 3), or RGBA (H x W x 4)
            outscale: Final output scale factor. Can differ from model's native scale
                     (will use bicubic resize for final adjustment)
            alpha_upsampler: Method for upscaling alpha channel: 'realesrgan' or 'bicubic'
        
        Returns:
            Tuple of (upscaled_image, img_mode):
            - upscaled_image: Enhanced image as numpy array in BGR format
            - img_mode: String indicating image mode ('RGB', 'RGBA', or 'L' for grayscale)
        
        Raises:
            ValueError: If input image is invalid (empty, wrong dimensions)
            RuntimeError: If model inference fails (e.g., OOM error)
        """
        # Input validation
        if image is None or image.size == 0:
            raise ValueError("Input image is empty or None")
        
        if len(image.shape) < 2 or len(image.shape) > 3:
            raise ValueError(f"Invalid image shape: {image.shape}. Expected 2D or 3D array")
        
        h_input, w_input = image.shape[:2]
        
        if h_input < 8 or w_input < 8:
            raise ValueError(f"Image too small: {w_input}x{h_input}. Minimum is 8x8 pixels")
        
        logger.debug(f"Upscaling image: {w_input}x{h_input} -> target scale {outscale}x")
        
        try:
            # Access upsampler (triggers lazy loading if needed)
            upsampler = self.upsampler
            
            # Perform inference
            # Note: enhance() expects BGR input and returns BGR output
            output, img_mode = upsampler.enhance(
                image,
                outscale=outscale,
                alpha_upsampler=alpha_upsampler
            )
            
            h_output, w_output = output.shape[:2]
            logger.info(f"Upscaling complete: {w_input}x{h_input} -> {w_output}x{h_output} ({img_mode})")
            
            return output, img_mode
            
        except RuntimeError as e:
            error_msg = str(e)
            logger.error(f"Real-ESRGAN inference failed: {error_msg}")
            
            # Provide helpful suggestions for common errors
            if "CUDA out of memory" in error_msg or "OOM" in error_msg:
                logger.error("GPU ran out of memory. Try:")
                logger.error("  - Reducing tile_size (e.g., 256 or 128)")
                logger.error("  - Using a smaller model (e.g., ANIME_VIDEO_V3)")
                logger.error("  - Closing other GPU applications")
                raise RuntimeError(
                    f"CUDA OOM: {error_msg}. Reduce tile_size or use CPU mode."
                ) from e
            
            raise RuntimeError(f"Inference failed: {error_msg}") from e
    
    def upscale_with_fallback(
        self,
        image: np.ndarray,
        outscale: float = 4.0,
        fallback_method: Literal['lanczos', 'cubic'] = 'lanczos'
    ) -> Tuple[np.ndarray, bool]:
        """
        Upscale with automatic fallback to traditional interpolation if SR fails.
        
        This method provides robust upscaling by attempting Real-ESRGAN first,
        then falling back to OpenCV interpolation if the model fails or isn't available.
        
        Args:
            image: Input image in BGR format
            outscale: Target scale factor
            fallback_method: Interpolation method for fallback: 'lanczos' or 'cubic'
        
        Returns:
            Tuple of (upscaled_image, used_sr):
            - upscaled_image: Enhanced image in BGR format
            - used_sr: Boolean indicating if SR was successful (True) or fallback used (False)
        """
        try:
            output, _ = self.upscale(image, outscale=outscale)
            return output, True
            
        except Exception as e:
            logger.warning(f"Real-ESRGAN failed, falling back to {fallback_method}: {e}")
            
            # Fallback to traditional interpolation
            h, w = image.shape[:2]
            new_w, new_h = int(w * outscale), int(h * outscale)
            
            if fallback_method == 'lanczos':
                output = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            else:  # cubic
                output = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            logger.info(f"Fallback upscaling complete: {w}x{h} -> {new_w}x{new_h}")
            return output, False
    
    def get_model_info(self) -> dict:
        """
        Get information about the current model configuration.
        
        Returns:
            Dictionary with model details:
            - model_type: The model type enum value
            - scale: Native upscaling factor
            - device: Current compute device
            - tile_size: Tiling configuration
            - half_precision: Whether FP16 is enabled
        """
        config = MODEL_CONFIGS.get(self.model_type, {})
        return {
            "model_type": self.model_type.value,
            "scale": config.get("scale", "unknown"),
            "architecture": config.get("arch", "unknown"),
            "device": str(self.device) if self._device else "not initialized",
            "tile_size": self.tile_size,
            "half_precision": self.use_half_precision,
            "model_loaded": self._upsampler is not None
        }
    
    def unload_model(self):
        """
        Unload the model from memory to free GPU/CPU resources.
        
        Useful when processing batches and wanting to release memory between runs,
        or when switching between different models.
        """
        if self._upsampler is not None:
            # Delete model references
            if hasattr(self._upsampler, 'model'):
                del self._upsampler.model
            self._upsampler = None
            self._model = None
            
            # Clear CUDA cache if using GPU
            if self._device and self._device.type == 'cuda':
                torch.cuda.empty_cache()
                logger.info("Model unloaded and CUDA cache cleared")
            else:
                logger.info("Model unloaded")
