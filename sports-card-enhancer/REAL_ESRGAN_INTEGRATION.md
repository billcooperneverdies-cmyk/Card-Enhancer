# Real-ESRGAN Integration Guide

## Overview

This document describes the integration of Real-ESRGAN super-resolution models into the Card-Enhancer project for high-quality image upscaling.

## Architecture

### Components

1. **`app/services/real_esrgan_service.py`** - Core service wrapper for Real-ESRGAN
   - Handles model loading and lifecycle management
   - Provides GPU/CPU device selection with automatic fallback
   - Implements tiled inference for large images
   - Supports multiple SR model architectures

2. **`app/services/enhancement_service.py`** - Main enhancement orchestrator
   - Integrates Real-ESRGAN into the existing enhancement pipeline
   - Provides fallback to traditional interpolation if SR fails
   - Exposes model selection via `SRModelChoice` enum

3. **`app/models/schemas.py`** - API schemas
   - `SRModelChoice` enum for model selection in API requests
   - Enhanced `EnhancementSettings` with `sr_model` field

## Supported Models

| Model | Scale | Best For | Description |
|-------|-------|----------|-------------|
| `REAL_ESRGAN_X4PLUS` | 4x | General photos | Best overall quality for sports cards |
| `REAL_ESRNET_X4PLUS` | 4x | Conservative | Less aggressive, fewer artifacts |
| `REAL_ESRGAN_ANIME_6B` | 4x | Illustrations | Optimized for anime/art style cards |
| `REAL_ESRGAN_X2PLUS` | 2x | Fast processing | Faster than 4x models |
| `ANIME_VIDEO_V3` | 4x | Lightweight | Smallest model, fastest inference |
| `GENERAL_X4V3` | 4x | Versatile | Includes denoise control |

## Usage

### Basic Usage

```python
from app.services.real_esrgan_service import RealESRGANService, SRModelType
import cv2

# Initialize service
service = RealESRGANService(
    model_type=SRModelType.REAL_ESRGAN_X4PLUS,
    tile_size=256  # Enable tiling for memory efficiency
)

# Load image (BGR format from OpenCV)
image = cv2.imread('card.jpg')

# Upscale
upscaled, mode = service.upscale(image, outscale=4.0)

# Save result
cv2.imwrite('card_upscaled.png', upscaled)
```

### With Fallback

```python
# Automatically falls back to Lanczos if SR fails
upscaled, used_sr = service.upscale_with_fallback(
    image, 
    outscale=4.0,
    fallback_method='lanczos'
)

if used_sr:
    print("Used Real-ESRGAN")
else:
    print("Used fallback interpolation")
```

### Via Enhancement Service

```python
from app.services.enhancement_service import EnhancementService
from app.models.schemas import EnhancementSettings, SRModelChoice

# Create service with specific model
service = EnhancementService(
    sr_model_choice=SRModelChoice.REAL_ESRGAN_X4PLUS
)

# Configure settings
settings = EnhancementSettings(
    upscaling=True,
    upscale_factor=4,
    sr_model=SRModelChoice.REAL_ESRGAN_X4PLUS,
    sharpening=True,
    color_correction=True
)

# Enhance image
enhanced_image, blemishes = service.enhance('card.jpg', settings)
```

## Data Flow

```
Input Image (BGR/OpenCV)
    ↓
EnhancementService.enhance()
    ↓
[Optional: Blemish Detection & Removal]
    ↓
[Optional: Noise Reduction]
    ↓
[Optional: Color Correction]
    ↓
[Optional: Contrast Enhancement]
    ↓
[Optional: Sharpening]
    ↓
Upscaling Requested?
    ├─→ Yes: RealESRGANService.upscale()
    │         ↓
    │     [Model Loading (lazy)]
    │         ↓
    │     [GPU/CPU Device Selection]
    │         ↓
    │     [Tiled Inference if needed]
    │         ↓
    │     Output (BGR/OpenCV)
    │
    └─→ No: Skip upscaling
    ↓
Output Image (BGR/OpenCV)
```

## Error Handling

### Common Issues and Solutions

1. **CUDA Out of Memory**
   - Reduce `tile_size` (e.g., 256 → 128)
   - Use a smaller model (`ANIME_VIDEO_V3`)
   - Close other GPU applications
   - Enable half precision (`use_half_precision=True`)

2. **Model Not Found**
   - Models are auto-downloaded on first use
   - Check internet connection
   - Verify write permissions in `weights/` directory

3. **Import Errors**
   ```bash
   pip install realesrgan basicsr gfpgan
   ```

4. **Slow Inference**
   - Ensure CUDA is available: `torch.cuda.is_available()`
   - Use half precision for faster inference
   - Consider smaller models for batch processing

## Configuration

### Environment Variables

```bash
# Optional: Specify GPU device
export CUDA_VISIBLE_DEVICES=0

# Optional: Custom model weights directory
export REAL_ESRGAN_WEIGHTS_PATH=/path/to/weights
```

### Tiling Configuration

For large images or limited GPU memory:

```python
service = RealESRGANService(
    tile_size=256,      # Process in 256x256 tiles
    tile_pad=10,        # 10px overlap between tiles
    pre_pad=0           # No border padding
)
```

Recommended tile sizes:
- 8GB+ VRAM: 512 or 0 (no tiling)
- 4-8GB VRAM: 256
- <4GB VRAM: 128 or use CPU mode

## Performance Benchmarks

Typical performance on NVIDIA RTX 3080 (10GB VRAM):

| Image Size | Model | Time (no tile) | Time (tile=256) |
|------------|-------|----------------|-----------------|
| 512x512 | REAL_ESRGAN_X4PLUS | ~0.5s | ~0.6s |
| 1024x1024 | REAL_ESRGAN_X4PLUS | ~2.0s | ~2.5s |
| 2048x2048 | REAL_ESRGAN_X4PLUS | OOM | ~8.0s |
| 1024x1024 | ANIME_VIDEO_V3 | ~0.8s | ~1.0s |

CPU-only performance (Intel i7-12700K):
- 5-10x slower than GPU
- Suitable for small batches or development

## Model Weights

Models are automatically downloaded from GitHub releases:
- Location: `sports-card-enhancer/weights/`
- Format: `.pth` (PyTorch)
- Sizes: 17MB - 67MB depending on model

Manual download URLs:
- RealESRGAN_x4plus: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
- RealESRNet_x4plus: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/RealESRNet_x4plus.pth
- RealESRGAN_x4plus_anime_6B: https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.2.4/RealESRGAN_x4plus_anime_6B.pth

## Dependencies

Required packages (see `requirements.txt`):
```
torch>=2.1.2
torchvision>=0.16.2
basicsr>=1.4.2
realesrgan>=0.3.0
gfpgan>=1.3.5  # optional
opencv-python>=4.9.0.76
```

Install with:
```bash
pip install -r backend/requirements.txt
```

## Testing

```python
# Test model loading
from app.services.real_esrgan_service import RealESRGANService, SRModelType

service = RealESRGANService(model_type=SRModelType.REAL_ESRGAN_X4PLUS)
info = service.get_model_info()
print(f"Model: {info['model_type']}")
print(f"Device: {info['device']}")
print(f"Loaded: {info['model_loaded']}")

# Test upscaling
import numpy as np
test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
result, used_sr = service.upscale_with_fallback(test_image, outscale=2.0)
assert result.shape == (512, 512, 3)
```

## Future Enhancements

- [ ] Support for custom trained models
- [ ] Batch processing optimizations
- [ ] Progressive upscaling (2x → 4x → 8x)
- [ ] Web UI model selector
- [ ] Model comparison tool
- [ ] Quantization for faster inference

## References

- Real-ESRGAN Repository: https://github.com/xinntao/Real-ESRGAN
- BasicSR Framework: https://github.com/XPixelGroup/BasicSR
- Paper: "Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data"
