"""Application configuration settings."""
from pydantic_settings import BaseSettings
from typing import Optional, Set
from pathlib import Path
import os


class Settings(BaseSettings):
    """Application configuration with environment variable support.
    
    Attributes:
        APP_NAME: Application name displayed in API docs
        APP_VERSION: Current version string
        DEBUG: Enable debug mode for development
        HOST: Server bind host
        PORT: Server bind port
        MAX_FILE_SIZE: Maximum single file size in bytes
        MAX_BATCH_SIZE: Maximum files per batch job
        ALLOWED_EXTENSIONS: Set of supported image extensions
        UPLOAD_DIR: Directory for uploaded files
        OUTPUT_DIR: Directory for processed outputs
        TEMP_DIR: Directory for temporary files
        MAX_IMAGE_DIMENSION: Maximum dimension for image resizing
        DEFAULT_OUTPUT_QUALITY: Default image quality (0-100)
        USE_GPU: Enable GPU acceleration if available
        MODEL_CACHE_DIR: Directory for cached ML models
        MAX_CONCURRENT_JOBS: Maximum parallel batch jobs
        JOB_TIMEOUT: Job timeout in seconds
    """
    
    # App settings
    APP_NAME: str = "Sports Card Enhancer API"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    
    # Server settings
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    
    # File upload settings
    MAX_FILE_SIZE: int = 50 * 1024 * 1024  # 50MB
    MAX_BATCH_SIZE: int = 100
    ALLOWED_EXTENSIONS: Set[str] = {".jpg", ".jpeg", ".png", ".tiff", ".tif", ".bmp"}
    
    # Storage settings
    BASE_DIR: Path = Path(".")
    UPLOAD_DIR: Path = Path("./uploads")
    OUTPUT_DIR: Path = Path("./outputs")
    TEMP_DIR: Path = Path("./temp")
    MODEL_CACHE_DIR: Path = Path("./models")
    
    # Processing settings
    MAX_IMAGE_DIMENSION: int = 4096
    DEFAULT_OUTPUT_QUALITY: int = 95
    
    # AI Model settings
    USE_GPU: bool = True
    
    # Batch processing
    MAX_CONCURRENT_JOBS: int = 4
    JOB_TIMEOUT: int = 300  # 5 minutes
    
    # External API tokens (optional)
    HUGGINGFACE_API_TOKEN: Optional[str] = None
    REPLICATE_API_TOKEN: Optional[str] = None
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    def ensure_directories(self) -> None:
        """Create all required directories if they don't exist."""
        for directory in [self.UPLOAD_DIR, self.OUTPUT_DIR, self.TEMP_DIR, self.MODEL_CACHE_DIR]:
            directory.mkdir(parents=True, exist_ok=True)
    
    @property
    def upload_dir_str(self) -> str:
        """Get upload directory as string."""
        return str(self.UPLOAD_DIR)
    
    @property
    def output_dir_str(self) -> str:
        """Get output directory as string."""
        return str(self.OUTPUT_DIR)
    
    @property
    def temp_dir_str(self) -> str:
        """Get temp directory as string."""
        return str(self.TEMP_DIR)


# Global settings instance
settings = Settings()
settings.ensure_directories()
