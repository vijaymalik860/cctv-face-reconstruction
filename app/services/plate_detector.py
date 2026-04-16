"""
Number Plate Detector Service

Detection pipeline (priority order):
  1. YOLO first — indian_license_plate.pt (highest accuracy)
     Tries: original, night-preprocessed, and lower-region crop
  2. OpenCV fallback — only if YOLO finds nothing
     a. White/Yellow plate HSV detector
     b. Adaptive threshold + contour (night-relaxed)
     c. Canny morph
     d. Dark plate edge detector (new — for night CCTV)

Night-vision:
  - Multi-stage CLAHE + gamma lift
  - Dehazing via dark channel prior
  - Color gates RELAXED for dark/underexposed images
  - Lower image region (bottom 60%) targeted scan
"""
import gc
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

from app.config import (
    MODEL_DIR,
    NIGHT_VISION_THRESHOLD,
    MAX_PLATES_PER_IMAGE,
)


class PlateDetector:
    def __init__(self):
        self._model        = None
        self._model_tried  = False
        self._models_dir   = Path(MODEL_DIR)
        print("[PlateDetector] Initialized (YOLO primary + OpenCV fallback)")

    # ------------------------------------------------------------------ #
    # Night-vision helpers                                                 #
    # ------------------------------------------------------------------ #

    def _is_night(self, image: np.ndarray) -> bool:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        return float(np.mean(gray)) < NIGHT_VISION_THRESHOLD

    def preprocess_night_vision(self, image: np.ndarray) -> np.ndarray:
        """Aggressive CLAHE + gamma + dehazing for very dark CCTV frames."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()

        # Stage 1: Strong CLAHE to bring out plate details
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(4, 4))
        enhanced = clahe.apply(gray)

        # Stage 2: Aggressive gamma lift (gamma=0.35 → very dark → bright)
        lut = np.array([min(255, int((i / 255.0) ** 0.35 * 255)) for i in range(256)], dtype=np.uint8)
        enhanced = cv2.LUT(enhanced, lut)

        # Stage 3: Second CLAHE pass to restore local contrast after gamma lift
        clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe2.apply(enhanced)

        bgr = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)

        # Stage 4: Simple dehazing — subtract dark channel blur
        try:
            dark = cv2.erode(bgr, np.ones((7, 7), np.uint8))
            dark_blur = cv2.GaussianBlur(dark, (21, 21), 0).astype(np.float32)
            bgr_f = bgr.astype(np.float32)
            dehazed = np.clip((bgr_f - dark_blur * 0.3 + 30), 0, 255).astype(np.uint8)
            return dehazed
        except Exception:
            return bgr

    # ------------------------------------------------------------------ #
    # Strict validation helpers                                            #
    # ------------------------------------------------------------------ #

    def _valid_plate_box(self, x: int, y: int, w: int, h: int,
                          iw: int, ih: int,
                          night: bool = False) -> bool:
        """
        License plate geometry check.
        Indian plates: ~330x110mm (3:1) or 200x100mm (2:1) for bikes.
        Night mode relaxes minimum size (dark CCTV plates appear small).
        """
        # Minimum real size — relaxed for night shots
        min_w = 40 if night else 60
        min_h = 12 if night else 18
        if w < min_w or h < min_h:
            return False
        # Don't take up too much of the image
        if w > iw * 0.92 or h > ih * 0.55:
            return False
        aspect = w / h if h > 0 else 0
        # Indian plates: aspect ratio 1.0 – 6.0 
        # (1.0 covers rare perfectly square bike/tractor plates, up to 6.0 for wide plates)
        lo, hi = (1.0, 6.5) if night else (1.0, 5.5)
        return lo <= aspect <= hi

    def _has_plate_colors(self, crop: np.ndarray,
                           night: bool = False) -> bool:
        """
        Color / texture gate for Indian license plates.

        Day mode: strict (white/yellow background, >=30% bright, >=6% dark text).
        Night mode: relaxed — dark CCTV plates have very low brightness;
                    we rely on texture entropy and reject only hard-red regions.
        """
        if crop is None or crop.size == 0:
            return False
        h, w = crop.shape[:2]
        if h < 8 or w < 20:
            return False

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop.copy()

        bright_ratio = float(np.sum(gray > 150)) / gray.size   # white background
        dark_ratio   = float(np.sum(gray < 70))  / gray.size   # text / dark pixels
        mid_ratio    = float(np.sum((gray >= 70) & (gray <= 150))) / gray.size

        # ── ALWAYS reject strongly red regions (tail-lights) ───────────
        if crop.ndim == 3:
            r_ch = crop[:, :, 2].astype(np.int16)
            g_ch = crop[:, :, 1].astype(np.int16)
            b_ch = crop[:, :, 0].astype(np.int16)
            red_mask = (r_ch > 100) & (r_ch * 10 > g_ch * 18) & (r_ch * 10 > b_ch * 18)
            del r_ch, g_ch, b_ch
            red_frac = float(red_mask.mean())
            del red_mask
            # Night: only kill very dominant red (>45% strongly red — real tail-light)
            # Day:   threshold stays at 15%
            red_limit = 0.45 if night else 0.15
            if red_frac > red_limit:
                return False

        # ── Texture entropy (works for day and night) ───────────────────
        _, counts = np.unique((gray // 16).ravel(), return_counts=True)
        probs   = counts / counts.sum()
        entropy = -float(np.sum(probs * np.log2(probs + 1e-12)))

        # ─────────────────────────────────────────────────────────────
        # NIGHT mode: plate will be dark — relax ALL brightness checks.
        # Accept if: reasonable texture (entropy>1.8) + any contrast
        # ─────────────────────────────────────────────────────────────
        if night:
            has_contrast = (dark_ratio >= 0.04 or bright_ratio >= 0.10 or
                            (dark_ratio + bright_ratio) >= 0.08)
            if entropy > 1.8 and has_contrast:
                return True
            # Night yellow plate (low saturation under sodium lamps)
            if crop.ndim == 3:
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                lower_y = np.array([10, 30, 30])
                upper_y = np.array([45, 255, 255])
                yellow_ratio = float(np.sum(cv2.inRange(hsv, lower_y, upper_y) > 0)) / gray.size
                if yellow_ratio >= 0.10 and entropy > 1.5:
                    return True
            return False

        # ─────────────────────────────────────────────────────────────
        # DAY mode: strict checks (now including EV Green & Rental Black)
        # ─────────────────────────────────────────────────────────────
        # 1. White plate (Private car)
        if bright_ratio >= 0.30 and dark_ratio >= 0.06 and entropy > 2.0:
            return True

        yellow_ratio = 0.0
        if crop.ndim == 3:
            hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
            
            # 2. Yellow plate (Commercial / Taxi)
            lower_y = np.array([15, 80, 100])
            upper_y = np.array([40, 255, 255])
            yellow_ratio = float(np.sum(cv2.inRange(hsv, lower_y, upper_y) > 0)) / gray.size
            if yellow_ratio >= 0.18 and dark_ratio >= 0.06:
                return True
                
            # 3. Green plate (Electric Vehicle - EV)
            lower_g = np.array([35, 40, 40])
            upper_g = np.array([85, 255, 255])
            green_ratio = float(np.sum(cv2.inRange(hsv, lower_g, upper_g) > 0)) / gray.size
            if green_ratio >= 0.15 and (bright_ratio >= 0.05 or yellow_ratio >= 0.03) and entropy > 1.6:
                return True

        # 4. Black plate (Rental / VIP / Army)
        # Very dark background with bright or yellow text
        if dark_ratio >= 0.40 and (bright_ratio >= 0.05 or yellow_ratio >= 0.03) and entropy > 1.5:
            return True

        return False

    def _nms(self, boxes: List[dict], iou_thresh: float = 0.4) -> List[dict]:
        """Confidence-based NMS."""
        if not boxes:
            return []
        boxes = sorted(boxes, key=lambda b: b["confidence"], reverse=True)
        kept = []
        for b in boxes:
            dup = False
            for k in kept:
                ox = max(0, min(b["x"] + b["w"], k["x"] + k["w"]) - max(b["x"], k["x"]))
                oy = max(0, min(b["y"] + b["h"], k["y"] + k["h"]) - max(b["y"], k["y"]))
                inter = ox * oy
                union = b["w"] * b["h"] + k["w"] * k["h"] - inter
                if union > 0 and inter / union > iou_thresh:
                    dup = True
                    break
            if not dup:
                kept.append(b)
        return kept

    # ================================================================== #
    # YOLO — PRIMARY detector                                              #
    # ================================================================== #

    def _load_yolo(self) -> bool:
        """Lazy-load YOLO model. Returns True if model is available."""
        if self._model_tried and self._model is None:
            return False
        if self._model is not None:
            return True

        self._model_tried = True
        candidates = [
            self._models_dir / "indian_license_plate.pt",
            self._models_dir / "yolov8n_license_plate.pt",
            self._models_dir / "yolov8_license_plate.pt",
        ]
        model_path: Optional[Path] = next((p for p in candidates if p.exists()), None)
        if model_path is None:
            print("[PlateDetector] No YOLO model found — OpenCV only mode")
            return False

        try:
            from ultralytics import YOLO
            self._model = YOLO(str(model_path))
            print(f"[PlateDetector] YOLO loaded: {model_path.name}")
            return True
        except Exception as e:
            print(f"[PlateDetector] YOLO load failed: {e}")
            self._model = None
            return False

    def _run_yolo(self, image: np.ndarray, is_night: bool) -> List[dict]:
        """
        Run YOLO with multiple source images:
          - Original
          - Night-preprocessed (CLAHE + gamma) if dark
          - Lower-region crop (bottom 65%) — number plates sit low on vehicles
          - Zoomed lower-region crop (2× zoom) — catches tiny/distant plates
        Coordinates from crops are remapped back to full-image space.
        """
        if not self._load_yolo():
            return []

        ih, iw = image.shape[:2]

        # Build list: (source_image, offset_x, offset_y, source_label)
        sources = [(image, 0, 0, "orig")]

        if is_night:
            nv = self.preprocess_night_vision(image)
            sources.append((nv, 0, 0, "night"))

        # Lower-region crop (bottom 65% of image — where plates live)
        lower_y = int(ih * 0.35)
        lower_crop = image[lower_y:, :]
        sources.append((lower_crop, 0, lower_y, "lower"))

        if is_night:
            nv_lower = self.preprocess_night_vision(lower_crop)
            sources.append((nv_lower, 0, lower_y, "night_lower"))

        # 2× zoom of lower-region — catches small distant plates
        lh, lw = lower_crop.shape[:2]
        zoomed = cv2.resize(lower_crop, (lw * 2, lh * 2), interpolation=cv2.INTER_CUBIC)
        sources.append((zoomed, 0, lower_y, "zoom2x"))  # coords will be halved back

        seen_keys: set = set()
        plates: List[dict] = []

        try:
            conf_thr = 0.10 if is_night else 0.15   # lower threshold at night

            for src, off_x, off_y, label in sources:
                sh, sw = src.shape[:2]
                # Scale factor from src back to original image
                sx = iw / sw if "zoom" in label else 1.0
                sy = image.shape[0] / (sh + off_y) if "zoom" in label else 1.0
                # For zoomed: sx=sy should be ~0.5 (half back)
                if "zoom" in label:
                    sx = lw / sw   # zoom scales width: sw = lw*2 → sx=0.5
                    sy = lh / sh   # same for height

                results = self._model(src, verbose=False, conf=conf_thr, iou=0.45)
                for r in results:
                    for box in r.boxes:
                        bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                        conf = float(box.conf[0])

                        # Map back to original image coordinates
                        x1 = int(bx1 * sx) + off_x
                        y1 = int(by1 * sy) + off_y
                        x2 = int(bx2 * sx) + off_x
                        y2 = int(by2 * sy) + off_y
                        w, h = x2 - x1, y2 - y1

                        key = (round(x1, -1), round(y1, -1), round(w, -1), round(h, -1))
                        if key in seen_keys:
                            continue
                        seen_keys.add(key)

                        if not self._valid_plate_box(x1, y1, w, h, iw, ih, night=is_night):
                            continue

                        # Validate colors (pass night flag so gates are relaxed)
                        crop = image[y1:y2, x1:x2]
                        if not self._has_plate_colors(crop, night=is_night):
                            continue

                        plates.append({"x": x1, "y": y1, "w": w, "h": h,
                                       "confidence": conf,
                                       "is_night_vision": is_night})

            if plates:
                print(f"[PlateDetector] YOLO found {len(plates)} plate(s) "
                      f"({'night' if is_night else 'day'})")
        except Exception as e:
            print(f"[PlateDetector] YOLO error: {e}")

        return plates

    # ================================================================== #
    # OpenCV fallback methods (used ONLY when YOLO finds nothing)         #
    # ================================================================== #

    def _opencv_white_plate(self, image: np.ndarray,
                              iw: int, ih: int, is_night: bool) -> List[dict]:
        """Detect bright white rectangles (Indian private vehicle plates)."""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        results: List[dict] = []
        # Night: include lower thresholds (80/100) to catch dim plates
        thresholds = [180, 155, 130, 100, 80] if is_night else [180, 155, 130]
        for thresh_val in thresholds:
            _, mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (28, 8))
            closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                if not self._valid_plate_box(x, y, w, h, iw, ih, night=is_night):
                    continue
                crop = image[y:y+h, x:x+w]
                if not self._has_plate_colors(crop, night=is_night):
                    continue
                conf = 0.78 - (180 - thresh_val) * 0.003
                results.append({"x": x, "y": y, "w": w, "h": h,
                                 "confidence": max(0.40, conf),
                                 "is_night_vision": is_night})
        return results

    def _opencv_yellow_plate(self, image: np.ndarray,
                               iw: int, ih: int, is_night: bool) -> List[dict]:
        """Detect yellow commercial vehicle plates using HSV."""
        if image.ndim != 3:
            return []
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Night: wider hue/saturation range (sodium street lights shift colors)
        if is_night:
            mask = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([45, 255, 255]))
        else:
            mask = cv2.inRange(hsv, np.array([15, 80, 80]), np.array([40, 255, 255]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (24, 8))
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[dict] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if not self._valid_plate_box(x, y, w, h, iw, ih, night=is_night):
                continue
            crop = image[y:y+h, x:x+w]
            if not self._has_plate_colors(crop, night=is_night):
                continue
            results.append({"x": x, "y": y, "w": w, "h": h,
                             "confidence": 0.80, "is_night_vision": is_night})
        return results

    def _opencv_dark_plate(self, image: np.ndarray,
                            iw: int, ih: int) -> List[dict]:
        """
        NEW: Detect dark/dim number plates in night CCTV using edge gradient.
        Looks for rectangular high-gradient regions in the lower image half.
        """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        # Focus on lower 65% (where plates are)
        y_start = int(ih * 0.30)
        roi = gray[y_start:, :]

        # Strong CLAHE to expose dark plate
        clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        roi_eq = clahe.apply(roi)

        # Sobel gradient magnitude
        sx = cv2.Sobel(roi_eq, cv2.CV_64F, 1, 0, ksize=3)
        sy = cv2.Sobel(roi_eq, cv2.CV_64F, 0, 1, ksize=3)
        grad = np.clip(np.sqrt(sx**2 + sy**2), 0, 255).astype(np.uint8)

        # Threshold gradient map
        _, gmask = cv2.threshold(grad, 25, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
        closed = cv2.morphologyEx(gmask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[dict] = []
        for cnt in contours:
            x, y_roi, w, h = cv2.boundingRect(cnt)
            y = y_roi + y_start  # remap to full image
            if not self._valid_plate_box(x, y, w, h, iw, ih, night=True):
                continue
            crop = image[y:y+h, x:x+w]
            if not self._has_plate_colors(crop, night=True):
                continue
            results.append({"x": x, "y": y, "w": w, "h": h,
                             "confidence": 0.52, "is_night_vision": True})
        return results

    def _opencv_canny(self, gray: np.ndarray, image: np.ndarray,
                       iw: int, ih: int, is_night: bool) -> List[dict]:
        """Canny edges + morphological close — general plate shape finder."""
        blur   = cv2.bilateralFilter(gray, 11, 17, 17)
        # Night: lower thresholds to catch faint edges
        lo, hi = (15, 120) if is_night else (30, 200)
        edges  = cv2.Canny(blur, lo, hi)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 5))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[dict] = []
        for cnt in contours:
            peri  = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.018 * peri, True)
            x, y, w, h = cv2.boundingRect(approx)
            if not self._valid_plate_box(x, y, w, h, iw, ih, night=is_night):
                continue
            crop = image[y:y+h, x:x+w]
            if not self._has_plate_colors(crop, night=is_night):
                continue
            results.append({"x": x, "y": y, "w": w, "h": h,
                             "confidence": 0.60, "is_night_vision": is_night})
        return results

    def _opencv_adaptive(self, gray: np.ndarray, image: np.ndarray,
                          iw: int, ih: int, is_night: bool) -> List[dict]:
        """Adaptive threshold fallback."""
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        # Night: larger block size catches low-contrast plate text
        block = 19 if is_night else 15
        thresh = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, block, 4
        )
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        results: List[dict] = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            if not self._valid_plate_box(x, y, w, h, iw, ih, night=is_night):
                continue
            crop = image[y:y+h, x:x+w]
            if not self._has_plate_colors(crop, night=is_night):
                continue
            results.append({"x": x, "y": y, "w": w, "h": h,
                             "confidence": 0.58, "is_night_vision": is_night})
        return results

    def _opencv_fallback(self, image: np.ndarray, is_night: bool) -> List[dict]:
        """Run all OpenCV methods as fallback. Only used when YOLO finds nothing.
        Note: `image` is already the downscaled work image."""
        ih, iw = image.shape[:2]
        inp    = self.preprocess_night_vision(image) if is_night else image
        gray   = cv2.cvtColor(inp, cv2.COLOR_BGR2GRAY)
        o_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        candidates: List[dict] = []
        candidates += self._opencv_white_plate(image, iw, ih, is_night)
        candidates += self._opencv_white_plate(inp,   iw, ih, is_night)
        candidates += self._opencv_yellow_plate(image, iw, ih, is_night)
        for src in [o_gray, gray]:
            candidates += self._opencv_canny(src,    image, iw, ih, is_night)
            candidates += self._opencv_adaptive(src, image, iw, ih, is_night)

        # Night: also run dark-plate gradient detector
        if is_night:
            candidates += self._opencv_dark_plate(image, iw, ih)
            candidates += self._opencv_dark_plate(inp, iw, ih)

        plates = self._nms(candidates, iou_thresh=0.35)
        print(f"[PlateDetector] OpenCV fallback: {len(plates)} plate(s)")
        return plates

    # ------------------------------------------------------------------ #
    # Main detect entry point                                              #
    # ------------------------------------------------------------------ #

    def detect_plates(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.15,
    ) -> List[dict]:
        """
        Detect number plates. Large images are scaled down to save RAM.
        YOLO runs first. OpenCV fallback only if YOLO returns nothing.
        Coordinates are rescaled back to the original image size.
        """
        # ── Scale down to save RAM (max 640px wide) ────────────────────
        MAX_DIM = 640
        oh, ow = image.shape[:2]
        scale = 1.0
        if ow > MAX_DIM or oh > MAX_DIM:
            scale = MAX_DIM / max(ow, oh)
            nw, nh = int(ow * scale), int(oh * scale)
            work = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        else:
            work = image

        is_night = self._is_night(work)

        # ── 1. Try YOLO (primary) ──────────────────────────────────────
        plates = self._run_yolo(work, is_night)

        # ── 2. OpenCV fallback if YOLO found nothing ───────────────────
        if not plates:
            print("[PlateDetector] YOLO empty — running OpenCV fallback")
            plates = self._opencv_fallback(work, is_night)

        # ── 3. Scale coordinates back to original size ─────────────────
        if scale != 1.0 and plates:
            inv = 1.0 / scale
            for p in plates:
                p["x"] = int(p["x"] * inv)
                p["y"] = int(p["y"] * inv)
                p["w"] = int(p["w"] * inv)
                p["h"] = int(p["h"] * inv)

        # Sort by confidence, cap results
        plates.sort(key=lambda p: p["confidence"], reverse=True)
        plates = plates[:MAX_PLATES_PER_IMAGE]

        print(f"[PlateDetector] Final: {len(plates)} plate(s) "
              f"({'night' if is_night else 'day'}) | scale={scale:.2f}")
        return plates

    # ------------------------------------------------------------------ #
    # Crop & Draw utilities                                                #
    # ------------------------------------------------------------------ #

    def crop_plates(
        self, image: np.ndarray, plates: List[dict], padding: int = 8
    ) -> List[np.ndarray]:
        ih, iw = image.shape[:2]
        crops: List[np.ndarray] = []
        for p in plates:
            w = p["w"]
            h = p["h"]
            
            # Unconditional Width Override: YOLO often returns tiny boxes of just the brightly 
            # lit half of a night plate, giving it high confidence while ignoring the shadowed half.
            # We force the crop width to a minimum aspect ratio of 5.0 (standard full plate).
            target_w = max(w, int(h * 5.0))
            pad_x = (target_w - w) // 2 + (padding * 2)
            pad_y = int(h * 0.25) + padding

            x  = max(0, p["x"] - pad_x)
            y  = max(0, p["y"] - pad_y)
            x2 = min(iw, p["x"] + w + pad_x)
            y2 = min(ih, p["y"] + h + pad_y)
            
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
                label += " [NV]"
            cv2.putText(out, label, (x, max(y - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 50), 2)
        return out


# Singleton
plate_detector = PlateDetector()
