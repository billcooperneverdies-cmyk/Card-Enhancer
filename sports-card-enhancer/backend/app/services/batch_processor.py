"""Batch processing service with job queue management."""
import asyncio
import uuid
import os
import zipfile
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Callable, List
from dataclasses import dataclass, field
import logging

from app.models.schemas import (
    JobStatus, BatchJob, CardImage, EnhancementSettings,
    WebSocketProgressMessage
)
from app.services.enhancement_service import EnhancementService
from app.core.config import settings
from app.utils.image_utils import ImageProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class Job:
    """Internal job representation."""
    id: str
    status: JobStatus
    created_at: datetime
    updated_at: datetime
    settings: EnhancementSettings
    image_paths: List[str]
    output_dir: str
    results: Dict = field(default_factory=dict)
    progress: float = 0.0
    error_message: Optional[str] = None
    total_processing_time_ms: int = 0


class BatchProcessor:
    """Manages batch processing jobs with async queue."""
    
    def __init__(self, max_concurrent: int = 4):
        self.max_concurrent = max_concurrent
        self.jobs: Dict[str, Job] = {}
        self.queue = asyncio.Queue()
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.enhancement_service = EnhancementService()
        self.image_processor = ImageProcessor()
        self.progress_callbacks: Dict[str, List[Callable]] = {}
        
        # Start worker task
        self.worker_task = None
    
    async def start(self):
        """Start the batch processor worker."""
        if self.worker_task is None:
            self.worker_task = asyncio.create_task(self._worker())
            logger.info("Batch processor started")
    
    async def stop(self):
        """Stop the batch processor."""
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass
            self.worker_task = None
            logger.info("Batch processor stopped")
    
    async def submit_job(self, image_paths: List[str], 
                        enhancement_settings: EnhancementSettings) -> str:
        """Submit a new batch job."""
        job_id = str(uuid.uuid4())
        output_dir = settings.OUTPUT_DIR / job_id
        output_dir.mkdir(parents=True, exist_ok=True)
        
        job = Job(
            id=job_id,
            status=JobStatus.PENDING,
            created_at=datetime.now(),
            updated_at=datetime.now(),
            settings=enhancement_settings,
            image_paths=image_paths,
            output_dir=str(output_dir)
        )
        
        self.jobs[job_id] = job
        await self.queue.put(job_id)
        
        logger.info(f"Job {job_id} submitted with {len(image_paths)} images")
        return job_id
    
    async def get_job(self, job_id: str) -> Optional[Job]:
        """Get job by ID."""
        return self.jobs.get(job_id)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel a pending or processing job."""
        job = self.jobs.get(job_id)
        if job and job.status in [JobStatus.PENDING, JobStatus.PROCESSING]:
            job.status = JobStatus.CANCELLED
            job.updated_at = datetime.now()
            return True
        return False
    
    def register_progress_callback(self, job_id: str, callback: Callable):
        """Register a callback for job progress updates."""
        if job_id not in self.progress_callbacks:
            self.progress_callbacks[job_id] = []
        self.progress_callbacks[job_id].append(callback)
    
    def unregister_progress_callback(self, job_id: str, callback: Callable):
        """Unregister a progress callback."""
        if job_id in self.progress_callbacks:
            self.progress_callbacks[job_id] = [
                cb for cb in self.progress_callbacks[job_id] 
                if cb != callback
            ]
    
    async def _worker(self):
        """Main worker loop that processes jobs from the queue."""
        while True:
            try:
                job_id = await self.queue.get()
                job = self.jobs.get(job_id)
                
                if job is None or job.status == JobStatus.CANCELLED:
                    self.queue.task_done()
                    continue
                
                # Process job with semaphore for concurrency control
                async with self.semaphore:
                    await self._process_job(job)
                
                self.queue.task_done()
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Worker error: {e}")
    
    async def _process_job(self, job: Job):
        """Process a single batch job."""
        start_time = datetime.now()
        job.status = JobStatus.PROCESSING
        job.updated_at = datetime.now()
        
        logger.info(f"Processing job {job.id}")
        
        try:
            total_images = len(job.image_paths)
            processed = 0
            failed = 0
            
            for i, image_path in enumerate(job.image_paths):
                # Check if job was cancelled
                if job.status == JobStatus.CANCELLED:
                    logger.info(f"Job {job.id} cancelled")
                    return
                
                try:
                    # Update progress
                    progress = (i / total_images) * 100
                    job.progress = progress
                    await self._notify_progress(job, i, total_images, 
                                               f"Processing {os.path.basename(image_path)}")
                    
                    # Process image
                    result = await self._process_single_image(job, image_path)
                    
                    if result['success']:
                        processed += 1
                    else:
                        failed += 1
                        logger.warning(f"Failed to process {image_path}: {result.get('error')}")
                
                except Exception as e:
                    failed += 1
                    logger.error(f"Error processing {image_path}: {e}")
            
            # Create ZIP archive if multiple images
            if total_images > 1:
                await self._create_zip_archive(job)
            
            # Update job status
            if failed == total_images:
                job.status = JobStatus.FAILED
                job.error_message = "All images failed to process"
            else:
                job.status = JobStatus.COMPLETED
            
            end_time = datetime.now()
            job.total_processing_time_ms = int((end_time - start_time).total_seconds() * 1000)
            job.progress = 100.0
            job.updated_at = datetime.now()
            
            await self._notify_progress(job, total_images, total_images, "Complete")
            
            logger.info(f"Job {job.id} completed: {processed} succeeded, {failed} failed")
        
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error_message = str(e)
            job.updated_at = datetime.now()
            logger.error(f"Job {job.id} failed: {e}")
    
    async def _process_single_image(self, job: Job, image_path: str) -> Dict:
        """Process a single image."""
        result = {
            'path': image_path,
            'success': False,
            'output_path': None,
            'error': None,
            'blemishes': []
        }
        
        try:
            # Enhance image
            enhanced_image, blemishes = self.enhancement_service.enhance(
                image_path, 
                job.settings
            )
            
            # Save enhanced image
            filename = os.path.basename(image_path)
            name, ext = os.path.splitext(filename)
            output_filename = f"{name}_enhanced.{job.settings.output_format}"
            output_path = Path(job.output_dir) / output_filename
            
            self.image_processor.save_image(
                enhanced_image, 
                str(output_path), 
                quality=job.settings.output_quality,
                dpi=job.settings.output_dpi
            )
            
            result['success'] = True
            result['output_path'] = str(output_path)
            result['blemishes'] = blemishes
            
            # Store in job results
            job.results[image_path] = result
        
        except Exception as e:
            result['error'] = str(e)
            job.results[image_path] = result
        
        return result
    
    async def _create_zip_archive(self, job: Job):
        """Create ZIP archive of all processed images."""
        zip_path = settings.OUTPUT_DIR / f"{job.id}.zip"
        
        with zipfile.ZipFile(str(zip_path), 'w', zipfile.ZIP_DEFLATED) as zipf:
            for root, dirs, files in os.walk(job.output_dir):
                for file in files:
                    file_path = Path(root) / file
                    arcname = file_path.relative_to(job.output_dir)
                    zipf.write(str(file_path), str(arcname))
        
        logger.info(f"Created ZIP archive: {zip_path}")
    
    async def _notify_progress(self, job: Job, current: int, total: int, message: str):
        """Notify progress callbacks."""
        progress = (current / total) * 100 if total > 0 else 0
        
        message_obj = WebSocketProgressMessage(
            job_id=job.id,
            image_id=None,
            status=job.status,
            progress=progress,
            message=message,
            timestamp=datetime.now()
        )
        
        # Call registered callbacks
        callbacks = self.progress_callbacks.get(job.id, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(message_obj)
                else:
                    callback(message_obj)
            except Exception as e:
                logger.error(f"Progress callback error: {e}")
    
    def to_batch_job(self, job: Job) -> BatchJob:
        """Convert internal Job to API BatchJob schema."""
        # Convert results to CardImage objects
        images = []
        for path, result in job.results.items():
            card = CardImage(
                id=str(uuid.uuid4()),
                filename=os.path.basename(path),
                original_path=path,
                processed_path=result.get('output_path'),
                width=0,  # Would need to load image to get dimensions
                height=0,
                format=os.path.splitext(path)[1].lower().replace('.', ''),
                size_bytes=os.path.getsize(path) if os.path.exists(path) else 0,
                status=JobStatus.COMPLETED if result['success'] else JobStatus.FAILED,
                progress=100.0 if result['success'] else 0.0,
                detected_blemishes=result.get('blemishes', []),
                error_message=result.get('error')
            )
            images.append(card)
        
        # Check for pending images
        for path in job.image_paths:
            if path not in job.results:
                card = CardImage(
                    id=str(uuid.uuid4()),
                    filename=os.path.basename(path),
                    original_path=path,
                    status=JobStatus.PENDING,
                    progress=0.0,
                    width=0,
                    height=0,
                    format=os.path.splitext(path)[1].lower().replace('.', ''),
                    size_bytes=os.path.getsize(path) if os.path.exists(path) else 0
                )
                images.append(card)
        
        # Determine ZIP path
        zip_path = None
        if len(job.image_paths) > 1:
            potential_zip = settings.OUTPUT_DIR / f"{job.id}.zip"
            if potential_zip.exists():
                zip_path = str(potential_zip)
        
        return BatchJob(
            id=job.id,
            status=job.status,
            created_at=job.created_at,
            updated_at=job.updated_at,
            total_images=len(job.image_paths),
            completed_images=sum(1 for r in job.results.values() if r['success']),
            failed_images=sum(1 for r in job.results.values() if not r['success']),
            settings=job.settings,
            images=images,
            output_zip_path=zip_path,
            total_processing_time_ms=job.total_processing_time_ms,
            error_message=job.error_message
        )


# Global batch processor instance
batch_processor = BatchProcessor(max_concurrent=settings.MAX_CONCURRENT_JOBS)
