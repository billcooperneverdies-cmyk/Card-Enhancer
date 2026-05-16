"""FastAPI main application."""
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os
import shutil
import zipfile
import io
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple
from datetime import datetime, timedelta
import json
import asyncio

from app.core.config import settings
from app.models.schemas import (
    UploadResponse, EnhancementResponse, JobStatusResponse, 
    DownloadResponse, EnhancementSettings, JobStatus, WebSocketProgressMessage
)
from app.services.batch_processor import batch_processor, Job
from app.utils.image_utils import ImageProcessor


def extract_zip_images(zip_content: bytes, extract_dir: Path) -> List[str]:
    """
    Extract images from a ZIP archive.
    
    Returns list of extracted image file paths.
    """
    extracted_paths = []
    allowed_extensions = {'.jpg', '.jpeg', '.png', '.tiff', '.tif', '.bmp', '.webp'}
    
    with zipfile.ZipFile(io.BytesIO(zip_content), 'r') as zip_ref:
        for file_info in zip_ref.infolist():
            # Skip directories and hidden files
            if file_info.is_dir() or file_info.filename.startswith('__MACOSX'):
                continue
            
            # Check extension
            filename = os.path.basename(file_info.filename)
            if filename.startswith('.'):
                continue
                
            ext = os.path.splitext(filename)[1].lower()
            if ext not in allowed_extensions:
                continue
            
            # Check file size (skip files > 50MB)
            if file_info.file_size > settings.MAX_FILE_SIZE:
                continue
            
            # Extract file
            try:
                # Create a safe filename (avoid path traversal)
                safe_filename = os.path.basename(filename)
                extract_path = extract_dir / safe_filename
                
                # Handle duplicate filenames
                base, ext = os.path.splitext(safe_filename)
                counter = 1
                while extract_path.exists():
                    extract_path = extract_dir / f"{base}_{counter}{ext}"
                    counter += 1
                
                with zip_ref.open(file_info) as source, open(extract_path, 'wb') as target:
                    shutil.copyfileobj(source, target)
                
                extracted_paths.append(str(extract_path))
            except Exception:
                continue
    
    return extracted_paths

# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="AI-powered sports card image enhancement API",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files for outputs
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=str(settings.OUTPUT_DIR)), name="outputs")

# Store active WebSocket connections
active_connections: dict = {}


@app.on_event("startup")
async def startup_event():
    """Start the batch processor on startup."""
    await batch_processor.start()


@app.on_event("shutdown")
async def shutdown_event():
    """Stop the batch processor on shutdown."""
    await batch_processor.stop()


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "endpoints": {
            "docs": "/docs",
            "upload": "/upload",
            "enhance": "/enhance",
            "status": "/status/{job_id}",
            "download": "/download/{job_id}",
            "preview": "/preview",
            "websocket": "/ws/{job_id}"
        }
    }


@app.post("/upload", response_model=UploadResponse)
async def upload_images(
    files: List[UploadFile] = File(...),
    settings_json: Optional[str] = Form(None)
):
    """
    Upload images for enhancement.
    
    - **files**: Single or multiple image files (.jpg, .png, .tiff, .bmp)
    - **settings_json**: Optional JSON string with enhancement settings
    """
    # Parse settings
    enhancement_settings = EnhancementSettings()
    if settings_json:
        try:
            settings_dict = json.loads(settings_json)
            enhancement_settings = EnhancementSettings(**settings_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid settings JSON: {e}")
    
    # Validate files
    accepted_files = []
    rejected_files = []
    rejected_reasons = []
    saved_paths = []
    
    for file in files:
        # Check file extension
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in settings.ALLOWED_EXTENSIONS:
            rejected_files.append(file.filename)
            rejected_reasons.append(f"{file.filename}: Unsupported format {ext}")
            continue
        
        # Check file size
        content = await file.read()
        if len(content) > settings.MAX_FILE_SIZE:
            rejected_files.append(file.filename)
            rejected_reasons.append(f"{file.filename}: File too large")
            continue
        
        # Save file
        try:
            upload_dir = Path(settings.UPLOAD_DIR) / job_id
            upload_dir.mkdir(parents=True, exist_ok=True)
            
            file_path = upload_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(content)
            
            accepted_files.append(file.filename)
            saved_paths.append(file_path)
        
        except Exception as e:
            rejected_files.append(file.filename)
            rejected_reasons.append(f"{file.filename}: Save failed - {str(e)}")
    
    if not accepted_files:
        raise HTTPException(status_code=400, detail="No valid files uploaded")
    
    return UploadResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Uploaded {len(accepted_files)} files successfully",
        total_files=len(files),
        accepted_files=len(accepted_files),
        rejected_files=len(rejected_files),
        rejected_reasons=rejected_reasons
    )


@app.post("/enhance", response_model=EnhancementResponse)
async def enhance_images(
    background_tasks: BackgroundTasks,
    files: List[UploadFile] = File(...),
    settings_json: Optional[str] = Form(None)
):
    """
    Upload and enhance images in one request.
    
    - **files**: Single or multiple image files, or ZIP archives containing images
    - **settings_json**: JSON string with enhancement settings
    """
    # Parse settings
    enhancement_settings = EnhancementSettings()
    if settings_json:
        try:
            settings_dict = json.loads(settings_json)
            enhancement_settings = EnhancementSettings(**settings_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid settings JSON: {e}")
    
    # Process and save uploaded files
    saved_paths = []
    temp_dir = settings.TEMP_DIR / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Extended allowed extensions including ZIP
    allowed_extensions = settings.ALLOWED_EXTENSIONS | {".zip"}
    
    for file in files:
        ext = os.path.splitext(file.filename)[1].lower()
        if ext not in allowed_extensions:
            continue
        
        content = await file.read()
        
        # Handle ZIP files
        if ext == ".zip":
            # Check ZIP size (allow up to 500MB for ZIPs)
            max_zip_size = settings.MAX_FILE_SIZE * 10
            if len(content) > max_zip_size:
                continue
            
            # Extract images from ZIP
            try:
                extracted_paths = extract_zip_images(content, temp_dir)
                saved_paths.extend(extracted_paths)
            except zipfile.BadZipFile:
                continue
            except Exception:
                continue
        else:
            # Regular image file
            if len(content) > settings.MAX_FILE_SIZE:
                continue
            
            file_path = temp_dir / file.filename
            with open(file_path, "wb") as f:
                f.write(content)
            
            saved_paths.append(str(file_path))
    
    if not saved_paths:
        raise HTTPException(status_code=400, detail="No valid files to process")
    
    # Limit batch size
    if len(saved_paths) > settings.MAX_BATCH_SIZE:
        saved_paths = saved_paths[:settings.MAX_BATCH_SIZE]
    
    # Submit batch job
    job_id = await batch_processor.submit_job(saved_paths, enhancement_settings)
    
    # Estimate processing time (rough estimate: 5 seconds per image)
    estimated_time = len(saved_paths) * 5
    
    return EnhancementResponse(
        job_id=job_id,
        status=JobStatus.PENDING,
        message=f"Enhancement job submitted with {len(saved_paths)} images",
        estimated_time_seconds=estimated_time
    )


@app.get("/status/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Get the status of a batch job."""
    job = await batch_processor.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    batch_job = batch_processor.to_batch_job(job)
    
    return JobStatusResponse(
        job_id=batch_job.id,
        status=batch_job.status,
        progress=job.progress,
        total_images=batch_job.total_images,
        completed_images=batch_job.completed_images,
        failed_images=batch_job.failed_images,
        images=batch_job.images,
        created_at=batch_job.created_at,
        updated_at=batch_job.updated_at
    )


@app.get("/download/{job_id}", response_model=DownloadResponse)
async def download_results(job_id: str):
    """Download enhanced images for a completed job."""
    job = await batch_processor.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if job.status != JobStatus.COMPLETED:
        raise HTTPException(status_code=400, detail=f"Job is not completed (status: {job.status})")
    
    # Determine download file
    if len(job.image_paths) > 1:
        # Return ZIP archive
        zip_path = settings.OUTPUT_DIR / f"{job_id}.zip"
        if zip_path.exists():
            download_url = f"/outputs/{job_id}.zip"
            total_size = zip_path.stat().st_size
            file_count = len(job.image_paths)
        else:
            raise HTTPException(status_code=500, detail="ZIP archive not found")
    else:
        # Return single file
        result = list(job.results.values())[0] if job.results else None
        if result and result.get('output_path'):
            output_path = Path(result['output_path'])
            filename = output_path.name
            download_url = f"/outputs/{job_id}/{filename}"
            total_size = output_path.stat().st_size
            file_count = 1
        else:
            raise HTTPException(status_code=500, detail="Output file not found")
    
    return DownloadResponse(
        job_id=job_id,
        download_url=download_url,
        expires_at=datetime.now() + timedelta(hours=24),
        total_size_bytes=total_size,
        file_count=file_count
    )


@app.get("/download/{job_id}/file")
async def download_file(job_id: str):
    """Direct file download endpoint."""
    job = await batch_processor.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    if len(job.image_paths) > 1:
        # Return ZIP
        zip_path = settings.OUTPUT_DIR / f"{job_id}.zip"
        if zip_path.exists():
            return FileResponse(
                str(zip_path),
                media_type="application/zip",
                filename=f"enhanced_cards_{job_id}.zip"
            )
    else:
        # Return single image
        result = list(job.results.values())[0] if job.results else None
        if result and result.get('output_path'):
            output_path = Path(result['output_path'])
            ext = output_path.suffix.lower()
            media_type = "image/png" if ext == ".png" else "image/jpeg"
            return FileResponse(
                str(output_path),
                media_type=media_type,
                filename=output_path.name
            )
    
    raise HTTPException(status_code=404, detail="File not found")


@app.post("/preview")
async def generate_preview(
    file: UploadFile = File(...),
    settings_json: Optional[str] = Form(None)
):
    """
    Generate a quick preview of enhancement settings.
    
    Returns a lower-quality preview for faster feedback.
    """
    # Parse settings
    enhancement_settings = EnhancementSettings()
    if settings_json:
        try:
            settings_dict = json.loads(settings_json)
            enhancement_settings = EnhancementSettings(**settings_dict)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid settings JSON: {e}")
    
    # Save uploaded file temporarily
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in settings.ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file format")
    
    content = await file.read()
    temp_filename = f"preview_{datetime.now().strftime('%Y%m%d%H%M%S')}{ext}"
    temp_path = settings.TEMP_DIR / temp_filename
    
    with open(temp_path, "wb") as f:
        f.write(content)
    
    try:
        # Generate preview
        from app.services.enhancement_service import EnhancementService
        service = EnhancementService()
        
        preview_image, blemishes = service.generate_preview(
            str(temp_path), 
            enhancement_settings,
            max_size=1024
        )
        
        # Save preview
        preview_path = temp_path.with_name(f"{temp_path.stem}_preview{ext}")
        ImageProcessor.save_image(preview_image, str(preview_path), quality=85)
        
        # Return file
        media_type = "image/png" if ext == ".png" else "image/jpeg"
        return FileResponse(
            str(preview_path),
            media_type=media_type,
            filename=f"preview_{file.filename}"
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Preview generation failed: {str(e)}")
    
    finally:
        # Cleanup temp files
        if temp_path.exists():
            temp_path.unlink()
        preview_path_candidate = temp_path.with_name(f"{temp_path.stem}_preview{ext}")
        if preview_path_candidate.exists():
            preview_path_candidate.unlink()


@app.websocket("/ws/{job_id}")
async def websocket_endpoint(websocket: WebSocket, job_id: str):
    """WebSocket endpoint for real-time job progress updates."""
    await websocket.accept()
    
    # Store connection
    if job_id not in active_connections:
        active_connections[job_id] = []
    active_connections[job_id].append(websocket)
    
    # Register progress callback
    async def progress_callback(message: WebSocketProgressMessage):
        await websocket.send_json(message.dict())
    
    batch_processor.register_progress_callback(job_id, progress_callback)
    
    try:
        # Send initial status
        job = await batch_processor.get_job(job_id)
        if job:
            await websocket.send_json({
                "job_id": job_id,
                "status": job.status.value,
                "progress": job.progress,
                "message": "Connected"
            })
        
        # Keep connection alive and handle client messages
        while True:
            data = await websocket.receive_text()
            message = json.loads(data)
            
            if message.get("action") == "cancel":
                success = await batch_processor.cancel_job(job_id)
                await websocket.send_json({
                    "job_id": job_id,
                    "action": "cancel",
                    "success": success
                })
    
    except WebSocketDisconnect:
        pass
    
    finally:
        # Cleanup
        batch_processor.unregister_progress_callback(job_id, progress_callback)
        if job_id in active_connections:
            active_connections[job_id] = [
                ws for ws in active_connections[job_id] 
                if ws != websocket
            ]


@app.delete("/jobs/{job_id}")
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    job = await batch_processor.get_job(job_id)
    
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Cancel if still processing
    if job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
        await batch_processor.cancel_job(job_id)
    
    # Remove output directory
    output_dir = Path(job.output_dir)
    if output_dir.exists():
        shutil.rmtree(output_dir)
    
    # Remove ZIP if exists
    zip_path = settings.OUTPUT_DIR / f"{job_id}.zip"
    if zip_path.exists():
        zip_path.unlink()
    
    # Remove from processor
    if job_id in batch_processor.jobs:
        del batch_processor.jobs[job_id]
    
    return {"message": f"Job {job_id} deleted successfully"}


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": settings.APP_VERSION
    }
