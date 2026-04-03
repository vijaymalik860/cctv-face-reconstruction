"""
Face Detection Service
Uses OpenCV's DNN-based face detector and Haar Cascade as fallback.
Detects faces in images, returns bounding boxes with confidence scores.
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
from app.config import OUTPUT_DIR


class FaceDetector:
    """Detects faces in images using OpenCV."""

    def __init__(self):
        self._dnn_net = None
        self._cascade = None
        self._init_detectors()

    def _init_detectors(self):
        """Initialize face detection models."""
        # Try DNN-based detector first (more accurate)
        try:
            prototxt_path = cv2.data.haarcascades + "../deploy.prototxt"
            model_path = cv2.data.haarcascades + "../res10_300x300_ssd_iter_140000.caffemodel"
            if Path(prototxt_path).exists() and Path(model_path).exists():
                self._dnn_net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                print("✅ DNN face detector loaded")
        except Exception:
            pass

        # Haar Cascade fallback (always available with OpenCV)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(cascade_path)
        if self._cascade.empty():
            # Try alternate path
            self._cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"
            )
        print("✅ Haar Cascade face detector loaded")

    def detect_faces(self, image: np.ndarray, min_confidence: float = 0.5) -> List[dict]:
        """
        Detect faces in an image.

        Args:
            image: BGR image as numpy array
            min_confidence: Minimum detection confidence (0-1)

        Returns:
            List of dicts with keys: x, y, w, h, confidence
        """
        if image is None:
            return []

        # Try DNN detector first
        if self._dnn_net is not None:
            faces = self._detect_dnn(image, min_confidence)
            if faces:
                return faces

        # Fallback to Haar Cascade
        return self._detect_cascade(image)

    def _detect_dnn(self, image: np.ndarray, min_confidence: float) -> List[dict]:
        """DNN-based face detection (SSD)."""
        h, w = image.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image, (300, 300)),
            1.0, (300, 300),
            (104.0, 177.0, 123.0)
        )
        self._dnn_net.setInput(blob)
        detections = self._dnn_net.forward()

        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > min_confidence:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype("int")
                # Ensure coordinates are valid
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if x2 > x1 and y2 > y1:
                    faces.append({
                        "x": int(x1),
                        "y": int(y1),
                        "w": int(x2 - x1),
                        "h": int(y2 - y1),
                        "confidence": float(confidence)
                    })
        return faces

    def _detect_cascade(self, image: np.ndarray) -> List[dict]:
        """Haar Cascade face detection (fallback)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        detections = self._cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )

        faces = []
        for (x, y, w, h) in detections:
            faces.append({
                "x": int(x),
                "y": int(y),
                "w": int(w),
                "h": int(h),
                "confidence": 0.85  # Cascade doesn't give confidence
            })
        return faces

    def crop_faces(
        self, image: np.ndarray, faces: List[dict],
        padding: float = 0.3, job_id: str = ""
    ) -> List[Tuple[np.ndarray, str]]:
        """
        Crop detected faces from image with padding.

        Args:
            image: Source image
            faces: List of face dicts from detect_faces()
            padding: Extra padding ratio around the face
            job_id: Job ID for naming files

        Returns:
            List of (cropped_image, saved_path) tuples
        """
        h, w = image.shape[:2]
        crops = []

        for i, face in enumerate(faces):
            fx, fy, fw, fh = face["x"], face["y"], face["w"], face["h"]

            # Add padding
            pad_w = int(fw * padding)
            pad_h = int(fh * padding)
            x1 = max(0, fx - pad_w)
            y1 = max(0, fy - pad_h)
            x2 = min(w, fx + fw + pad_w)
            y2 = min(h, fy + fh + pad_h)

            crop = image[y1:y2, x1:x2].copy()

            # Save cropped face
            face_dir = OUTPUT_DIR / "faces"
            face_dir.mkdir(parents=True, exist_ok=True)
            crop_path = face_dir / f"{job_id}_face_{i}_original.jpg"
            cv2.imwrite(str(crop_path), crop)
            crops.append((crop, str(crop_path)))

        return crops

    def draw_faces(self, image: np.ndarray, faces: List[dict]) -> np.ndarray:
        """Draw bounding boxes on detected faces."""
        annotated = image.copy()
        for i, face in enumerate(faces):
            x, y, w, h = face["x"], face["y"], face["w"], face["h"]
            conf = face.get("confidence", 0)

            # Draw rectangle
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Add label
            label = f"Face {i+1}: {conf:.0%}"
            cv2.putText(
                annotated, label, (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2
            )

        return annotated


# Singleton instance
face_detector = FaceDetector()
