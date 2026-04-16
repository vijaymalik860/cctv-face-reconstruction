"""
Number Plate Rectifier Service

Straightens tilted / perspective-warped plate crops so OCR reads them cleanly.

Pipeline:
  1. WPOD-NET (if model files exist) — precise perspective correction
  2. Fallback: simple resize to TARGET_W × TARGET_H

TensorFlow is lazy-loaded only on first use to avoid MemoryError at startup.
"""
import os
import sys
import cv2
import numpy as np

from app.config import MODEL_DIR

# Suppress TensorFlow logging (set before any tf import)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Import local_utils for WPOD-NET (lazy — only needed if model is present)
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
try:
    from local_utils import detect_lp
except ImportError:
    detect_lp = None


class PlateRectifier:
    def __init__(self):
        self._model = None
        self._models_dir = MODEL_DIR
        print("[PlateRectifier] Initialized with WPOD-NET")

    def _load_model(self):
        if self._model is not None:
            return

        json_path = os.path.join(self._models_dir, "wpod_net.json")
        h5_path = os.path.join(self._models_dir, "wpod_net.h5")

        if not os.path.exists(json_path) or not os.path.exists(h5_path):
            print("[PlateRectifier] WPOD-NET model files not found!")
            return

        try:
            # Lazy import — TF only loaded when WPOD-NET model files are present
            import tensorflow as tf
            from tensorflow.keras.models import model_from_json

            with open(json_path, 'r') as f:
                json_str = f.read()
            self._model = model_from_json(json_str, custom_objects={'tf': tf})
            self._model.load_weights(h5_path)
            print("[PlateRectifier] WPOD-NET Loaded")
        except Exception as e:
            print(f"[PlateRectifier] Failed to load WPOD-NET: {e}")

    def rectify(self, plate_crop: np.ndarray) -> np.ndarray:
        """
        Uses WPOD-NET to find the 4 corners of the plate inside the crop 
        and rectifies (straightens) it.
        """
        if plate_crop is None or plate_crop.size == 0:
            return np.full((100, 300, 3), 128, dtype=np.uint8)

        # Base fallback if anything fails
        fallback_img = cv2.resize(plate_crop, (300, 100), interpolation=cv2.INTER_LANCZOS4)

        if detect_lp is None:
            print("[PlateRectifier] local_utils.detect_lp not found, using fallback")
            return fallback_img

        self._load_model()
        if self._model is None:
            return fallback_img

        try:
            # Extreme illumination equalization before WPOD-NET to prevent polygon from chopping shadowed text
            if plate_crop.ndim == 3:
                lab = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2LAB)
                l_channel, a, b = cv2.split(lab)
                clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
                cl = clahe.apply(l_channel)
                lab = cv2.merge((cl, a, b))
                ench_crop = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                
                # Apply aggressive gamma lift to blast dark shadows bright (gamma ~0.35)
                lut = np.array(
                    [min(255, int((i / 255.0) ** 0.35 * 255)) for i in range(256)],
                    dtype=np.uint8
                )
                ench_crop = cv2.LUT(ench_crop, lut)
            else:
                ench_crop = plate_crop

            # WPOD-NET typically expects BGR image converted to RGB and normalized
            img_rgb = cv2.cvtColor(ench_crop, cv2.COLOR_BGR2RGB)
            img_norm = img_rgb.astype(np.float32) / 255.0

            ratio = float(max(img_norm.shape[:2])) / min(img_norm.shape[:2])
            side = int(ratio * 288.)
            bound_dim = min(side, 608)

            # Detect plate polygon and warp
            _, TLp, _, _ = detect_lp(self._model, img_norm, bound_dim, lp_threshold=0.5)

            if len(TLp) > 0:
                # highest confidence plate
                plate_rgb_norm = TLp[0]
                plate_rgb = (plate_rgb_norm * 255.0).astype(np.uint8)
                plate_bgr = cv2.cvtColor(plate_rgb, cv2.COLOR_RGB2BGR)
                return cv2.resize(plate_bgr, (300, 100), interpolation=cv2.INTER_LANCZOS4)
            else:
                # No polygon found by WPOD-NET inside this crop, run fallback
                return fallback_img
        except Exception as e:
            print(f"[PlateRectifier] WPOD-NET Error: {e}")
            return fallback_img

# Singleton
plate_rectifier = PlateRectifier()
