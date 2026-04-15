"""
Number Plate Detector Service
Pipeline:
  Stage 1 — YOLOv8n detects vehicles (car/truck/bus/motorcycle) from COCO weights
  Stage 2 — Haar Cascade searches for number plate INSIDE each vehicle crop
  Fallback — Haar Cascade on full image if Stage 1 finds no vehicles

Night-vision:
  - Auto-detects IR/low-light frames by mean brightness
  - CLAHE + bilateral filter applied before detection
"""
import gc
import os
import cv2
import numpy as np
import urllib.request
from pathlib import Path
from typing import List

from app.config import (
    MODEL_DIR,
    NIGHT_VISION_THRESHOLD,
    MAX_PLATES_PER_IMAGE,
)

# License plate specific YOLOv8 model
_YOLO_NAME = "yolov8n_license_plate.pt"

class PlateDetector:
    def __init__(self):
        self._model      = None
        self._models_dir = Path(MODEL_DIR)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        print("[PlateDetector] Initialized")

    # ------------------------------------------------------------------ #
    # Night-vision helpers                                                 #
    # ------------------------------------------------------------------ #

    def _is_night(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(np.mean(gray)) < NIGHT_VISION_THRESHOLD

    def preprocess_night_vision(self, image: np.ndarray) -> np.ndarray:
        """CLAHE + bilateral filter on grayscale, returned as BGR."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        gray = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        gray = cv2.equalizeHist(gray)
        gray = cv2.bilateralFilter(gray, 9, 75, 75)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------ #
    # Model loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_yolo(self):
        if self._model is not None:
            return
        path = self._models_dir / _YOLO_NAME
        if not path.exists():
            print(f"[PlateDetector] {_YOLO_NAME} not found. Please ensure it is in the models dir.")
            raise FileNotFoundError(f"{_YOLO_NAME} not found")
        from ultralytics import YOLO
        self._model = YOLO(str(path))
        print("[PlateDetector] YOLO license plate detector loaded")

    # ------------------------------------------------------------------ #
    # Detection                                                            #
    # ------------------------------------------------------------------ #

    def detect_plates(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
    ) -> List[dict]:
        """
        Returns list of dicts: {x, y, w, h, confidence, is_night_vision}
        Coordinates are on the ORIGINAL (un-preprocessed) image.
        """
        is_night = self._is_night(image)
        inp      = self.preprocess_night_vision(image) if is_night else image

        try:
            self._load_yolo()
        except Exception as e:
            print(f"[PlateDetector] YOLO load failed ({e})")
            return []

        gc.collect()

        results = self._model(inp, conf=conf_threshold, verbose=False)
        plates = []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                # For license plate model, class 0 is usually the plate
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                plates.append({
                    "x": int(x1),
                    "y": int(y1),
                    "w": int(x2) - int(x1),
                    "h": int(y2) - int(y1),
                    "confidence": float(box.conf[0]),
                    "is_night_vision": is_night,
                })

        plates = plates[:MAX_PLATES_PER_IMAGE]
        print(f"[PlateDetector] {len(plates)} plate(s) "
              f"({'night' if is_night else 'normal'})")
        return plates

    # ------------------------------------------------------------------ #
    # Haar Cascade helpers                                                 #
    # ------------------------------------------------------------------ #

    def _cascade_in_crop(self, crop: np.ndarray, is_night: bool) -> List[dict]:
        """Run Haar Cascade inside a vehicle crop. Returns crop-relative boxes."""
        if crop is None or crop.size == 0:
            return []
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 10)
        )
        result = []
        iw, ih = crop.shape[1], crop.shape[0]
        for (x, y, w, h) in (rects if len(rects) else []):
            aspect = w / h if h > 0 else 0
            if 1.8 <= aspect <= 7.5 and (w * h) / (iw * ih) <= 0.45:
                result.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                                "confidence": 0.70, "is_night_vision": is_night})
        return result

    def _cascade_full(self, image: np.ndarray, is_night: bool) -> List[dict]:
        """Run Haar Cascade on the full image."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        rects = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 20)
        )
        result = []
        for (x, y, w, h) in (rects if len(rects) else []):
            if 1.8 <= (w / h if h > 0 else 0) <= 7.5:
                result.append({"x": int(x), "y": int(y), "w": int(w), "h": int(h),
                                "confidence": 0.5, "is_night_vision": is_night})
        return result

    # ------------------------------------------------------------------ #
    # Crop & Draw utilities                                                #
    # ------------------------------------------------------------------ #

    def crop_plates(
        self, image: np.ndarray, plates: List[dict], padding: int = 5
    ) -> List[np.ndarray]:
        ih, iw = image.shape[:2]
        crops = []
        for p in plates:
            x  = max(0, p["x"] - padding)
            y  = max(0, p["y"] - padding)
            x2 = min(iw, p["x"] + p["w"] + padding)
            y2 = min(ih, p["y"] + p["h"] + padding)
            c  = image[y:y2, x:x2].copy()
            if c.size > 0:
                crops.append(c)
        return crops

    def draw_plates(self, image: np.ndarray, plates: List[dict]) -> np.ndarray:
        out = image.copy()
        for i, p in enumerate(plates):
            x, y, w, h = p["x"], p["y"], p["w"], p["h"]
            cv2.rectangle(out, (x, y), (x + w, y + h), (0, 220, 50), 2)
            label = f"Plate {i+1} ({p['confidence']:.2f})"
            if p.get("is_night_vision"):
                label += " [IR]"
            cv2.putText(out, label, (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 50), 2)
        return out


# Singleton
plate_detector = PlateDetector()
