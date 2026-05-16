"""
Test suite for Real-ESRGAN integration.
"""
import pytest
import numpy as np
import cv2
import tempfile
import os


class TestSRModelType:
    def test_enum_values(self):
        from app.services.real_esrgan_service import SRModelType
        assert SRModelType.REAL_ESRGAN_X4PLUS.value == "RealESRGAN_x4plus"
        assert SRModelType.ANIME_VIDEO_V3.value == "realesr-animevideov3"


class TestRealESRGANService:
    def test_service_initialization(self):
        from app.services.real_esrgan_service import RealESRGANService, SRModelType
        service = RealESRGANService(model_type=SRModelType.REAL_ESRGAN_X4PLUS, tile_size=256)
        assert service.model_type == SRModelType.REAL_ESRGAN_X4PLUS
        assert service._upsampler is None
    
    def test_fallback_upscaling(self):
        from app.services.real_esrgan_service import RealESRGANService
        service = RealESRGANService()
        test_img = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        result, used_sr = service.upscale_with_fallback(test_img, outscale=2.0)
        assert result.shape == (128, 128, 3)
        assert used_sr is False


class TestEnhancementSettingsWithSR:
    def test_default_sr_model(self):
        from app.models.schemas import EnhancementSettings, SRModelChoice
        settings = EnhancementSettings()
        assert settings.sr_model == SRModelChoice.REAL_ESRGAN_X4PLUS


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
