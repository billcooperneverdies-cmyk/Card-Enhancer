"""Typed schemas for the /api/v1 card analysis surface (Pydantic v2)."""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


class ArtifactLayer(str, Enum):
    ORIGINAL_CAPTURE = "ORIGINAL_CAPTURE"
    ORIENTATION_NORMALIZED = "ORIENTATION_NORMALIZED"
    PERSPECTIVE_RECTIFIED = "PERSPECTIVE_RECTIFIED"
    ANALYSIS_OVERLAY = "ANALYSIS_OVERLAY"
    PROMPTIR_RESTORED = "PROMPTIR_RESTORED"
    GFPGAN_FACE_RESTORED = "GFPGAN_FACE_RESTORED"


class AnalysisJobState(str, Enum):
    QUEUED = "QUEUED"
    RUNNING = "RUNNING"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    RETRY_PENDING = "RETRY_PENDING"
    CANCELLED = "CANCELLED"


class ImageArtifact(BaseModel):
    """Immutable image lineage record. Originals are never overwritten."""

    artifact_id: UUID
    parent_artifact_id: Optional[UUID] = None
    layer: ArtifactLayer
    storage_key: str
    width: int
    height: int
    created_at: datetime
    transform: dict[str, Any] = Field(default_factory=dict)
    model_name: Optional[str] = None
    model_version: Optional[str] = None


class CropResult(BaseModel):
    corners: list[list[float]]  # 4 x [x, y] in original-image coordinates
    confidence: float
    homography: list[list[float]]
    output_width: int
    output_height: int
    warnings: list[str] = Field(default_factory=list)


class OCRLine(BaseModel):
    text: str
    normalized_text: str
    confidence: float
    bounding_box: list[list[float]]


class OCRResult(BaseModel):
    status: str  # "ok" | "failed" | "unavailable"
    engine: str
    model_version: Optional[str] = None
    lines: list[OCRLine] = Field(default_factory=list)
    error: Optional[str] = None


class DCPTResult(BaseModel):
    status: str  # "ok" | "failed" | "unavailable"
    defect_logits: Optional[list[float]] = None
    defect_probabilities: Optional[list[float]] = None
    value_estimate: Optional[float] = None
    model_version: Optional[str] = None
    device: Optional[str] = None
    latency_ms: Optional[float] = None
    checkpoint_loaded: bool = False
    warnings: list[str] = Field(default_factory=list)
    error: Optional[str] = None


class CardAnalysis(BaseModel):
    crop: Optional[CropResult] = None
    ocr: Optional[OCRResult] = None
    dcpt: Optional[DCPTResult] = None


class AnalysisJob(BaseModel):
    job_id: UUID
    card_id: UUID
    state: AnalysisJobState
    progress: float = 0.0
    stage: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class Card(BaseModel):
    card_id: UUID
    original_filename: str
    created_at: datetime
    artifacts: list[ImageArtifact] = Field(default_factory=list)
    analysis: Optional[CardAnalysis] = None
    job: Optional[AnalysisJob] = None


class CardCreatedResponse(BaseModel):
    card_id: UUID
    job_id: UUID
    state: AnalysisJobState
    message: str


class ReadyResponse(BaseModel):
    ready: bool
    checks: dict[str, str]
