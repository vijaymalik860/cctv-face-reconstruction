"""
API Routes for Face Enhancement
Handles image/video upload, AI enhancement, job management, and history.
"""
import os
import uuid
import time
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.config import (
    UPLOAD_DIR, OUTPUT_DIR,
    ALLOWED_IMAGE_TYPES, ALLOWED_VIDEO_TYPES, MAX_FILE_SIZE
)
from app.models.schemas import ProcessingJob, FaceRegion
from app.services.face_detector import face_detector
from app.services.enhancer import face_enhancer
from app.services.video_processor import video_processor

router = APIRouter(prefix="/api", tags=["Enhancement"])


# ============================================================
# Image Enhancement
# ============================================================

@router.post("/enhance/image")
async def enhance_image(
    file: UploadFile = File(...),
    model: str = Form("gfpgan"),
    upscale: int = Form(2),
    face_enhance: bool = Form(True),
    bg_enhance: bool = Form(True),
    db: Session = Depends(get_db)
):
    """Upload and enhance a single image."""

    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported file type: {ext}. Allowed: {ALLOWED_IMAGE_TYPES}")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded file
    upload_path = UPLOAD_DIR / f"{job_id}{ext}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(400, f"File too large. Max size: {MAX_FILE_SIZE // (1024*1024)}MB")
        f.write(content)

    # Create job record
    job = ProcessingJob(
        id=uuid.UUID(job_id),
        original_filename=file.filename,
        input_path=str(upload_path),
        model_used=model,
        upscale_factor=upscale,
        status="processing",
        file_type="image"
    )
    db.add(job)
    db.commit()

    try:
        # Read image from saved file (more reliable than imdecode)
        import gc
        gc.collect()
        image = cv2.imread(str(upload_path))

        if image is None:
            raise ValueError("Could not decode image")

        # Detect faces
        faces = face_detector.detect_faces(image)
        job.faces_detected = len(faces)

        # Save annotated image (with face boxes)
        if faces:
            annotated = face_detector.draw_faces(image, faces)
            annotated_path = OUTPUT_DIR / f"{job_id}_annotated.jpg"
            cv2.imwrite(str(annotated_path), annotated)
            del annotated
            gc.collect()

        # Crop and save original face regions
        face_crops = face_detector.crop_faces(image, faces, job_id=job_id)

        # Enhance image
        enhanced_img, cropped_faces, restored_faces, proc_time = face_enhancer.enhance_image(
            image, model=model, upscale=upscale,
            face_enhance=face_enhance, bg_enhance=bg_enhance
        )

        # Save results
        save_results = face_enhancer.save_results(
            job_id, enhanced_img, cropped_faces, restored_faces
        )

        # Save face regions to database
        for i, face in enumerate(faces):
            face_region = FaceRegion(
                job_id=uuid.UUID(job_id),
                x=face["x"], y=face["y"],
                w=face["w"], h=face["h"],
                confidence=face["confidence"],
                cropped_path=face_crops[i][1] if i < len(face_crops) else None,
                enhanced_path=(
                    save_results["face_paths"][i]["enhanced"]
                    if i < len(save_results.get("face_paths", []))
                    else None
                )
            )
            db.add(face_region)

        # Update job record
        job.output_path = save_results["enhanced_path"]
        job.processing_time = proc_time
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()

        # Build response
        response = {
            "job_id": job_id,
            "status": "completed",
            "original_filename": file.filename,
            "model_used": model,
            "upscale_factor": upscale,
            "faces_detected": len(faces),
            "processing_time": round(proc_time, 2),
            "original_url": f"/files/uploads/{job_id}{ext}",
            "enhanced_url": f"/files/outputs/{job_id}_enhanced.jpg",
            "faces": []
        }

        for i, face in enumerate(faces):
            face_data = {
                "id": i,
                "x": face["x"], "y": face["y"],
                "w": face["w"], "h": face["h"],
                "confidence": round(face["confidence"], 3),
                "cropped_url": f"/files/outputs/faces/{job_id}_face_{i}_original.jpg",
                "enhanced_url": f"/files/outputs/faces/{job_id}_face_{i}_enhanced.jpg"
            }
            response["faces"].append(face_data)

        return JSONResponse(response)

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(500, f"Enhancement failed: {str(e)}")


# ============================================================
# Video Enhancement
# ============================================================

@router.post("/enhance/video")
async def enhance_video(
    file: UploadFile = File(...),
    model: str = Form("gfpgan"),
    upscale: int = Form(2),
    frame_interval: int = Form(5),
    max_frames: int = Form(50),
    db: Session = Depends(get_db)
):
    """Upload and enhance a video (frame-by-frame)."""

    # Validate file type
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported video type: {ext}. Allowed: {ALLOWED_VIDEO_TYPES}")

    # Generate job ID
    job_id = str(uuid.uuid4())

    # Save uploaded video
    upload_path = UPLOAD_DIR / "videos" / f"{job_id}{ext}"
    with open(upload_path, "wb") as f:
        content = await file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(400, "Video too large. Max size: 100MB")
        f.write(content)

    # Create job record
    job = ProcessingJob(
        id=uuid.UUID(job_id),
        original_filename=file.filename,
        input_path=str(upload_path),
        model_used=model,
        upscale_factor=upscale,
        status="processing",
        file_type="video"
    )
    db.add(job)
    db.commit()

    try:
        # Get video info
        video_info = video_processor.get_video_info(str(upload_path))

        # Process video
        results = video_processor.enhance_video(
            str(upload_path), job_id,
            model=model, upscale=upscale,
            frame_interval=frame_interval,
            max_frames=max_frames
        )

        # Update job
        job.output_path = results["output_path"]
        job.faces_detected = results["total_faces_detected"]
        job.processing_time = results["processing_time"]
        job.status = "completed"
        job.completed_at = datetime.utcnow()
        db.commit()

        response = {
            "job_id": job_id,
            "status": "completed",
            "original_filename": file.filename,
            "model_used": model,
            "video_info": video_info,
            "frames_processed": results["frames_processed"],
            "total_faces_detected": results["total_faces_detected"],
            "processing_time": results["processing_time"],
            "original_url": f"/files/uploads/videos/{job_id}{ext}",
            "enhanced_url": f"/files/outputs/videos/{job_id}_enhanced.mp4"
        }

        return JSONResponse(response)

    except Exception as e:
        job.status = "failed"
        job.error_message = str(e)
        job.completed_at = datetime.utcnow()
        db.commit()
        raise HTTPException(500, f"Video enhancement failed: {str(e)}")


# ============================================================
# Job Management
# ============================================================

@router.get("/jobs")
async def list_jobs(
    limit: int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    file_type: Optional[str] = None,
    db: Session = Depends(get_db)
):
    """List all processing jobs with optional filters."""
    query = db.query(ProcessingJob)

    if status:
        query = query.filter(ProcessingJob.status == status)
    if file_type:
        query = query.filter(ProcessingJob.file_type == file_type)

    total = query.count()
    jobs = query.order_by(ProcessingJob.created_at.desc()).offset(offset).limit(limit).all()

    return {
        "jobs": [_job_to_response(j) for j in jobs],
        "total": total,
        "limit": limit,
        "offset": offset
    }


@router.get("/jobs/{job_id}")
async def get_job(job_id: str, db: Session = Depends(get_db)):
    """Get details of a specific job."""
    job = db.query(ProcessingJob).filter(
        ProcessingJob.id == uuid.UUID(job_id)
    ).first()

    if not job:
        raise HTTPException(404, "Job not found")

    return _job_to_response(job)


@router.delete("/jobs/{job_id}")
async def delete_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a job and its associated files."""
    job = db.query(ProcessingJob).filter(
        ProcessingJob.id == uuid.UUID(job_id)
    ).first()

    if not job:
        raise HTTPException(404, "Job not found")

    # Delete files
    for path in [job.input_path, job.output_path]:
        if path and os.path.exists(path):
            os.remove(path)

    # Delete face region files
    for face in job.faces:
        for path in [face.cropped_path, face.enhanced_path]:
            if path and os.path.exists(path):
                os.remove(path)

    # Delete frame directories
    frames_dir = OUTPUT_DIR / "frames" / job_id
    if frames_dir.exists():
        shutil.rmtree(str(frames_dir))

    enhanced_frames_dir = OUTPUT_DIR / "frames" / f"{job_id}_enhanced"
    if enhanced_frames_dir.exists():
        shutil.rmtree(str(enhanced_frames_dir))

    # Delete from database
    db.delete(job)
    db.commit()

    return {"message": "Job deleted successfully", "job_id": job_id}


# ============================================================
# Statistics
# ============================================================

@router.get("/stats")
async def get_stats(db: Session = Depends(get_db)):
    """Get processing statistics."""
    total_jobs = db.query(ProcessingJob).count()
    completed_jobs = db.query(ProcessingJob).filter(
        ProcessingJob.status == "completed"
    ).count()
    total_faces = db.query(func.sum(ProcessingJob.faces_detected)).scalar() or 0
    avg_time = db.query(func.avg(ProcessingJob.processing_time)).filter(
        ProcessingJob.status == "completed"
    ).scalar()

    return {
        "total_jobs": total_jobs,
        "completed_jobs": completed_jobs,
        "failed_jobs": db.query(ProcessingJob).filter(
            ProcessingJob.status == "failed"
        ).count(),
        "total_faces_detected": int(total_faces),
        "avg_processing_time": round(float(avg_time), 2) if avg_time else 0,
        "models_info": face_enhancer.get_model_info()
    }


# ============================================================
# Model Info
# ============================================================

@router.get("/models")
async def get_models():
    """Get available AI models and their info."""
    return {
        "models": [
            {
                "id": "gfpgan",
                "name": "GFPGAN v1.4",
                "description": "Best for blind face restoration. Restores eyes, nose, mouth with realistic textures.",
                "supports_face_enhance": True,
                "supports_bg_enhance": True,
                "recommended": True
            },
            {
                "id": "realesrgan",
                "name": "Real-ESRGAN",
                "description": "Background and full image upscaling. Best for overall quality improvement.",
                "supports_face_enhance": False,
                "supports_bg_enhance": True,
                "recommended": False
            },
            {
                "id": "fsrcnn",
                "name": "Fast SR (Low RAM)",
                "description": "Lightweight super-resolution AI. Excellent for low-end CPUs and insufficient virtual memory scenarios.",
                "supports_face_enhance": False,
                "supports_bg_enhance": True,
                "recommended": False
            }
        ],
        "device": face_enhancer.get_model_info()["device"]
    }


# ============================================================
# Helpers
# ============================================================

def _job_to_response(job: ProcessingJob) -> dict:
    """Convert a ProcessingJob ORM object to API response dict."""
    job_id = str(job.id)
    ext = Path(job.original_filename).suffix.lower()

    response = {
        "id": job_id,
        "original_filename": job.original_filename,
        "model_used": job.model_used,
        "upscale_factor": job.upscale_factor,
        "faces_detected": job.faces_detected or 0,
        "processing_time": round(job.processing_time, 2) if job.processing_time else None,
        "status": job.status,
        "file_type": job.file_type,
        "error_message": job.error_message,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "completed_at": job.completed_at.isoformat() if job.completed_at else None,
    }

    if job.file_type == "image":
        response["original_url"] = f"/files/uploads/{job_id}{ext}"
        response["enhanced_url"] = f"/files/outputs/{job_id}_enhanced.jpg" if job.output_path else None
    elif job.file_type == "video":
        response["original_url"] = f"/files/uploads/videos/{job_id}{ext}"
        response["enhanced_url"] = f"/files/outputs/videos/{job_id}_enhanced.mp4" if job.output_path else None

    # Include face data
    response["faces"] = []
    if job.faces:
        for i, face in enumerate(job.faces):
            response["faces"].append({
                "id": str(face.id),
                "x": face.x, "y": face.y,
                "w": face.w, "h": face.h,
                "confidence": round(face.confidence, 3) if face.confidence else 0,
                "cropped_url": f"/files/outputs/faces/{job_id}_face_{i}_original.jpg",
                "enhanced_url": f"/files/outputs/faces/{job_id}_face_{i}_enhanced.jpg"
            })

    return response
