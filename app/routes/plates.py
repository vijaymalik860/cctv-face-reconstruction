"""
API Routes — Vehicle Number Plate Detection & Enhancement
Pipeline: YOLOv8 (detect) → WPOD-NET (rectify) → Real-ESRGAN (sharpen) → PaddleOCR (read)

Endpoints:
    POST   /api/plates/detect         — image upload  → full pipeline
    POST   /api/plates/detect/video   — video upload  → frame-by-frame
    GET    /api/plates/jobs           — list all plate jobs
    GET    /api/plates/jobs/{id}      — single job detail
    DELETE /api/plates/jobs/{id}      — delete job + files
    GET    /api/plates/stats          — aggregate statistics
"""
import os
import gc
import uuid
import time
import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, UploadFile, Form, Depends, HTTPException
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.database import get_db
from app.config import (
    UPLOAD_DIR, OUTPUT_DIR,
    ALLOWED_IMAGE_TYPES, ALLOWED_VIDEO_TYPES,
    MAX_FILE_SIZE, PLATE_UPSCALE,
)
from app.models.schemas import PlateDetectionJob, DetectedPlate
from app.services.plate_detector  import plate_detector
from app.services.plate_rectifier import plate_rectifier
from app.services.plate_ocr       import plate_ocr
from app.services.enhancer        import face_enhancer

router = APIRouter(prefix="/api/plates", tags=["Number Plates"])


# ============================================================
# Helpers
# ============================================================

def _plate_output_dir(job_id: str) -> Path:
    """Per-job output directory for plate files."""
    d = OUTPUT_DIR / "plates" / job_id
    d.mkdir(parents=True, exist_ok=True)
    return d


def _run_pipeline(
    image: np.ndarray,
    job_id: str,
    db: Session,
    job: PlateDetectionJob,
    upscale: int = PLATE_UPSCALE,
    force_night: Optional[bool] = None,
    conf_threshold: float = 0.25,
    ocr_threshold: float = 0.0,
) -> list:
    """
    Core plate pipeline for a single image frame.
    Returns a list of DetectedPlate ORM objects (not yet committed).
    """
    out_dir = _plate_output_dir(job_id)

    # ---- Step 1: Detect ----
    if force_night is True:
        image = plate_detector.preprocess_night_vision(image)
    plates_meta = plate_detector.detect_plates(image, conf_threshold=conf_threshold)

    # Save annotated original
    if plates_meta:
        annotated = plate_detector.draw_plates(image, plates_meta)
        cv2.imwrite(str(out_dir / "annotated.jpg"), annotated)
        del annotated
        gc.collect()

    plate_records = []
    crops = plate_detector.crop_plates(image, plates_meta)

    for i, (meta, crop) in enumerate(zip(plates_meta, crops)):
        prefix = f"plate_{i}"

        # Save raw crop
        raw_path = out_dir / f"{prefix}_raw.jpg"
        cv2.imwrite(str(raw_path), crop)

        # ---- Step 2: Rectify ----
        rectified = plate_rectifier.rectify(crop)
        rect_path = out_dir / f"{prefix}_rectified.jpg"
        cv2.imwrite(str(rect_path), rectified)

        # ---- Step 3: Enhance ----
        enhanced = face_enhancer.enhance_plate(rectified, upscale=upscale)
        enh_path  = out_dir / f"{prefix}_enhanced.jpg"
        cv2.imwrite(str(enh_path), enhanced)

        # ---- Step 4: OCR ----
        ocr_result = plate_ocr.read_text(enhanced)
        
        # Apply OCR Threshold filter
        final_ocr_text = ocr_result["text"] if ocr_result["confidence"] >= ocr_threshold else ""
        
        # Save per-plate metadata JSON
        with open(out_dir / f"{prefix}_meta.json", "w") as f:
            json.dump({
                "bbox": {"x": meta["x"], "y": meta["y"],
                         "w": meta["w"], "h": meta["h"]},
                "detection_confidence": meta["confidence"],
                "is_night_vision": meta["is_night_vision"],
                "ocr_text":       final_ocr_text,
                "ocr_confidence": ocr_result["confidence"],
            }, f, indent=2)

        # Build ORM record
        plate_record = DetectedPlate(
            job_id               = uuid.UUID(job_id),
            bbox_x               = meta["x"],
            bbox_y               = meta["y"],
            bbox_w               = meta["w"],
            bbox_h               = meta["h"],
            detection_confidence = meta["confidence"],
            is_night_vision      = meta["is_night_vision"],
            original_crop_path   = str(raw_path),
            rectified_path       = str(rect_path),
            enhanced_path        = str(enh_path),
            ocr_text             = final_ocr_text,
            ocr_confidence       = ocr_result["confidence"],
        )
        plate_records.append(plate_record)

        # Free memory after each plate
        del crop, rectified, enhanced
        gc.collect()

    return plate_records


def _build_plate_response(plate: DetectedPlate, job_id: str, idx: int) -> dict:
    """Convert DetectedPlate ORM record to API response dict."""
    prefix = f"plate_{idx}"
    return {
        "id":                   str(plate.id),
        "bbox":                 {"x": plate.bbox_x, "y": plate.bbox_y,
                                 "w": plate.bbox_w, "h": plate.bbox_h},
        "detection_confidence": round(plate.detection_confidence or 0, 3),
        "is_night_vision":      plate.is_night_vision,
        "ocr_text":             plate.ocr_text or "",
        "ocr_confidence":       round(plate.ocr_confidence or 0, 3),
        "original_crop_url":    f"/files/plates/{job_id}/{prefix}_raw.jpg",
        "rectified_url":        f"/files/plates/{job_id}/{prefix}_rectified.jpg",
        "enhanced_url":         f"/files/plates/{job_id}/{prefix}_enhanced.jpg",
        "annotated_url":        f"/files/plates/{job_id}/annotated.jpg",
    }


def _job_to_response(job: PlateDetectionJob) -> dict:
    """Convert PlateDetectionJob ORM record to API response dict."""
    job_id = str(job.id)
    ext    = Path(job.original_filename).suffix.lower()

    response = {
        "id":               job_id,
        "original_filename": job.original_filename,
        "file_type":        job.file_type,
        "status":           job.status,
        "plates_detected":  job.plates_detected or 0,
        "processing_time":  round(job.processing_time, 2) if job.processing_time else None,
        "error_message":    job.error_message,
        "created_at":       job.created_at.isoformat() if job.created_at else None,
        "completed_at":     job.completed_at.isoformat() if job.completed_at else None,
        "original_url":     f"/files/uploads/plates/{job_id}{ext}",
        "annotated_url":    f"/files/plates/{job_id}/annotated.jpg",
        "plates":           [],
    }

    if job.plates:
        for i, plate in enumerate(job.plates):
            response["plates"].append(_build_plate_response(plate, job_id, i))

    return response


# ============================================================
# POST /api/plates/detect  — Image
# ============================================================

@router.post("/detect")
async def detect_plates_image(
    file:        UploadFile = File(...),
    upscale:     int  = Form(PLATE_UPSCALE),
    force_night: bool = Form(False),
    conf_threshold: float = Form(0.25),
    ocr_threshold: float = Form(0.0),
    db: Session = Depends(get_db),
):
    """Upload an image and run the full number plate detection pipeline."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(400, f"Unsupported type: {ext}. Allowed: {ALLOWED_IMAGE_TYPES}")

    job_id    = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / "plates" / f"{job_id}{ext}"

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "File too large (max 100 MB)")
    with open(upload_path, "wb") as f:
        f.write(content)

    # Create job
    job = PlateDetectionJob(
        id=uuid.UUID(job_id),
        original_filename=file.filename,
        input_path=str(upload_path),
        file_type="image",
        status="processing",
    )
    db.add(job)
    db.commit()

    start = time.time()
    try:
        image = cv2.imread(str(upload_path))
        if image is None:
            raise ValueError("Could not read image")

        plate_records = _run_pipeline(
            image, job_id, db, job,
            upscale=upscale,
            force_night=force_night if force_night else None,
            conf_threshold=conf_threshold,
            ocr_threshold=ocr_threshold,
        )

        for rec in plate_records:
            db.add(rec)

        job.plates_detected = len(plate_records)
        job.processing_time = time.time() - start
        job.status          = "completed"
        job.completed_at    = datetime.utcnow()
        db.commit()

        response = _job_to_response(job)
        return JSONResponse(response)

    except Exception as e:
        job.status        = "failed"
        job.error_message = str(e)
        job.completed_at  = datetime.utcnow()
        db.commit()
        raise HTTPException(500, f"Plate detection failed: {e}")


# ============================================================
# POST /api/plates/detect/video  — Video (frame-by-frame)
# ============================================================

@router.post("/detect/video")
async def detect_plates_video(
    file:           UploadFile = File(...),
    upscale:        int  = Form(PLATE_UPSCALE),
    frame_interval: int  = Form(10),   # process every Nth frame
    max_frames:     int  = Form(30),   # cap total frames processed
    force_night:    bool = Form(False),
    conf_threshold: float = Form(0.25),
    ocr_threshold: float = Form(0.0),
    db: Session = Depends(get_db),
):
    """Upload a video and detect number plates frame-by-frame."""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(400, f"Unsupported video type: {ext}")

    job_id      = str(uuid.uuid4())
    upload_path = UPLOAD_DIR / "plates" / f"{job_id}{ext}"

    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(400, "Video too large (max 100 MB)")
    with open(upload_path, "wb") as f:
        f.write(content)

    job = PlateDetectionJob(
        id=uuid.UUID(job_id),
        original_filename=file.filename,
        input_path=str(upload_path),
        file_type="video",
        status="processing",
    )
    db.add(job)
    db.commit()

    start = time.time()
    try:
        cap = cv2.VideoCapture(str(upload_path))
        if not cap.isOpened():
            raise ValueError("Cannot open video file")

        frame_idx       = 0
        frames_processed = 0
        all_plate_records = []

        while frames_processed < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                try:
                    records = _run_pipeline(
                        frame, job_id, db, job,
                        upscale=upscale,
                        force_night=force_night if force_night else None,
                        conf_threshold=conf_threshold,
                        ocr_threshold=ocr_threshold,
                    )
                    all_plate_records.extend(records)
                    frames_processed += 1
                except Exception as frame_err:
                    print(f"[plates] Frame {frame_idx} failed: {frame_err}")
                finally:
                    del frame
                    gc.collect()

            frame_idx += 1

        cap.release()

        for rec in all_plate_records:
            db.add(rec)

        job.plates_detected = len(all_plate_records)
        job.processing_time = time.time() - start
        job.status          = "completed"
        job.completed_at    = datetime.utcnow()
        db.commit()

        response               = _job_to_response(job)
        response["frames_processed"] = frames_processed
        return JSONResponse(response)

    except Exception as e:
        job.status        = "failed"
        job.error_message = str(e)
        job.completed_at  = datetime.utcnow()
        db.commit()
        raise HTTPException(500, f"Video plate detection failed: {e}")


# ============================================================
# GET /api/plates/jobs
# ============================================================

@router.get("/jobs")
async def list_plate_jobs(
    limit:  int = 20,
    offset: int = 0,
    status: Optional[str] = None,
    db: Session = Depends(get_db),
):
    """List all plate detection jobs (newest first)."""
    query = db.query(PlateDetectionJob)
    if status:
        query = query.filter(PlateDetectionJob.status == status)

    total = query.count()
    jobs  = query.order_by(PlateDetectionJob.created_at.desc()) \
                 .offset(offset).limit(limit).all()

    return {
        "jobs":   [_job_to_response(j) for j in jobs],
        "total":  total,
        "limit":  limit,
        "offset": offset,
    }


# ============================================================
# GET /api/plates/jobs/{id}
# ============================================================

@router.get("/jobs/{job_id}")
async def get_plate_job(job_id: str, db: Session = Depends(get_db)):
    """Get details of a single plate detection job."""
    job = db.query(PlateDetectionJob).filter(
        PlateDetectionJob.id == uuid.UUID(job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Plate job not found")
    return _job_to_response(job)


# ============================================================
# DELETE /api/plates/jobs/{id}
# ============================================================

@router.delete("/jobs/{job_id}")
async def delete_plate_job(job_id: str, db: Session = Depends(get_db)):
    """Delete a plate job, its DB records, and all associated files."""
    job = db.query(PlateDetectionJob).filter(
        PlateDetectionJob.id == uuid.UUID(job_id)
    ).first()
    if not job:
        raise HTTPException(404, "Plate job not found")

    # Delete input file
    if job.input_path and os.path.exists(job.input_path):
        os.remove(job.input_path)

    # Delete entire per-job output directory
    out_dir = OUTPUT_DIR / "plates" / job_id
    if out_dir.exists():
        shutil.rmtree(str(out_dir))

    db.delete(job)
    db.commit()
    return {"message": "Plate job deleted", "job_id": job_id}


# ============================================================
# GET /api/plates/stats
# ============================================================

@router.get("/stats")
async def get_plate_stats(db: Session = Depends(get_db)):
    """Return aggregate stats for the plate detection module."""
    total     = db.query(PlateDetectionJob).count()
    completed = db.query(PlateDetectionJob).filter(
        PlateDetectionJob.status == "completed"
    ).count()
    failed    = db.query(PlateDetectionJob).filter(
        PlateDetectionJob.status == "failed"
    ).count()
    total_plates = db.query(
        func.sum(PlateDetectionJob.plates_detected)
    ).scalar() or 0
    avg_time = db.query(
        func.avg(PlateDetectionJob.processing_time)
    ).filter(PlateDetectionJob.status == "completed").scalar()

    # Night-vision count from DetectedPlate table
    night_count = db.query(DetectedPlate).filter(
        DetectedPlate.is_night_vision == True  # noqa: E712
    ).count()

    return {
        "total_jobs":         total,
        "completed_jobs":     completed,
        "failed_jobs":        failed,
        "total_plates_detected": int(total_plates),
        "night_vision_plates":   night_count,
        "avg_processing_time":   round(float(avg_time), 2) if avg_time else 0,
    }
