"""
Video Processing Service
Extracts frames from CCTV video, enhances each frame, and reconstructs output video.
Uses OpenCV for frame manipulation.
"""
import cv2
import time
import uuid
import numpy as np
from pathlib import Path
from typing import Optional, Callable

from app.config import OUTPUT_DIR, UPLOAD_DIR
from app.services.face_detector import face_detector
from app.services.enhancer import face_enhancer


class VideoProcessor:
    """Handles CCTV video frame extraction, enhancement, and reconstruction."""

    def __init__(self):
        print("🎥 VideoProcessor initialized")

    def get_video_info(self, video_path: str) -> dict:
        """Get metadata about a video file."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {"error": "Could not open video file"}

        info = {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            "duration_seconds": 0
        }
        if info["fps"] > 0:
            info["duration_seconds"] = round(info["total_frames"] / info["fps"], 2)

        cap.release()
        return info

    def extract_frames(
        self, video_path: str, job_id: str,
        frame_interval: int = 1, max_frames: Optional[int] = None
    ) -> list:
        """
        Extract frames from a video.

        Args:
            video_path: Path to the video file
            job_id: Unique job identifier
            frame_interval: Extract every Nth frame (1 = all frames)
            max_frames: Maximum number of frames to extract

        Returns:
            List of paths to extracted frames
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames_dir = OUTPUT_DIR / "frames" / job_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        frame_count = 0
        saved_count = 0

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"📹 Extracting frames from video ({total_frames} total)...")

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % frame_interval == 0:
                frame_path = frames_dir / f"frame_{saved_count:06d}.jpg"
                cv2.imwrite(str(frame_path), frame)
                frame_paths.append(str(frame_path))
                saved_count += 1

                if max_frames and saved_count >= max_frames:
                    break

            frame_count += 1

        cap.release()
        print(f"✅ Extracted {saved_count} frames from {frame_count} total")
        return frame_paths

    def enhance_video(
        self, video_path: str, job_id: str,
        model: str = "gfpgan", upscale: int = 2,
        frame_interval: int = 5, max_frames: Optional[int] = 100,
        progress_callback: Optional[Callable] = None
    ) -> dict:
        """
        Full video enhancement pipeline:
        1. Extract frames
        2. Detect and enhance faces in each frame
        3. Reconstruct enhanced video

        Args:
            video_path: Input video path
            job_id: Unique job ID
            model: AI model to use
            upscale: Upscale factor
            frame_interval: Process every Nth frame
            max_frames: Limit number of frames
            progress_callback: Optional callback(current, total, status)

        Returns:
            Dict with results including output path, stats
        """
        start_time = time.time()
        results = {
            "output_path": "",
            "frames_processed": 0,
            "total_faces_detected": 0,
            "processing_time": 0,
            "enhanced_frames": []
        }

        # Get video info
        video_info = self.get_video_info(video_path)
        if "error" in video_info:
            raise ValueError(video_info["error"])

        # Extract frames
        if progress_callback:
            progress_callback(0, 100, "Extracting frames...")

        frame_paths = self.extract_frames(
            video_path, job_id, frame_interval, max_frames
        )

        if not frame_paths:
            raise ValueError("No frames extracted from video")

        # Enhance each frame
        enhanced_frames_dir = OUTPUT_DIR / "frames" / f"{job_id}_enhanced"
        enhanced_frames_dir.mkdir(parents=True, exist_ok=True)

        total = len(frame_paths)
        for i, frame_path in enumerate(frame_paths):
            try:
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue

                # Detect faces
                faces = face_detector.detect_faces(frame)
                results["total_faces_detected"] += len(faces)

                # Enhance frame
                enhanced, cropped, restored, proc_time = face_enhancer.enhance_image(
                    frame, model=model, upscale=upscale
                )

                # Save enhanced frame
                enhanced_path = enhanced_frames_dir / f"enhanced_{i:06d}.jpg"
                cv2.imwrite(str(enhanced_path), enhanced)
                results["enhanced_frames"].append(str(enhanced_path))
                results["frames_processed"] += 1

                if progress_callback:
                    progress = int((i + 1) / total * 100)
                    progress_callback(
                        progress, 100,
                        f"Enhanced frame {i+1}/{total} ({len(faces)} faces)"
                    )

                print(f"  Frame {i+1}/{total} enhanced ({len(faces)} faces, {proc_time:.1f}s)")

            except Exception as e:
                print(f"  ⚠️ Frame {i+1} failed: {e}")
                continue

        # Reconstruct video
        if results["enhanced_frames"]:
            if progress_callback:
                progress_callback(95, 100, "Reconstructing video...")

            output_video_path = self._reconstruct_video(
                results["enhanced_frames"],
                job_id,
                fps=video_info["fps"] / frame_interval if frame_interval > 1 else video_info["fps"]
            )
            results["output_path"] = output_video_path

        results["processing_time"] = round(time.time() - start_time, 2)

        if progress_callback:
            progress_callback(100, 100, "Complete!")

        print(f"🎬 Video enhancement complete: {results['frames_processed']} frames, "
              f"{results['total_faces_detected']} faces, {results['processing_time']}s")

        return results

    def _reconstruct_video(
        self, frame_paths: list, job_id: str,
        fps: float = 24.0
    ) -> str:
        """Reconstruct video from enhanced frames."""
        if not frame_paths:
            return ""

        # Read first frame to get dimensions
        first_frame = cv2.imread(frame_paths[0])
        if first_frame is None:
            return ""

        h, w = first_frame.shape[:2]

        # Output path
        output_dir = OUTPUT_DIR / "videos"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{job_id}_enhanced.mp4"

        # Create video writer
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(output_path), fourcc, max(fps, 1.0), (w, h)
        )

        for fp in frame_paths:
            frame = cv2.imread(fp)
            if frame is not None:
                # Resize if needed (ensure consistent dimensions)
                if frame.shape[:2] != (h, w):
                    frame = cv2.resize(frame, (w, h))
                writer.write(frame)

        writer.release()
        print(f"✅ Video reconstructed: {output_path}")
        return str(output_path)

    def extract_key_frames(
        self, video_path: str, job_id: str,
        num_frames: int = 10
    ) -> list:
        """
        Extract key frames evenly distributed across the video.
        Useful for quick preview / face extraction.
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return []

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []

        interval = max(1, total // num_frames)
        frames_dir = OUTPUT_DIR / "frames" / job_id
        frames_dir.mkdir(parents=True, exist_ok=True)

        frame_paths = []
        for i in range(num_frames):
            frame_idx = min(i * interval, total - 1)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                path = frames_dir / f"keyframe_{i:04d}.jpg"
                cv2.imwrite(str(path), frame)
                frame_paths.append(str(path))

        cap.release()
        return frame_paths


# Singleton instance
video_processor = VideoProcessor()
