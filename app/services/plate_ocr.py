"""
Number Plate OCR Service

Uses PaddleOCR (PP-OCRv4) to read text from a rectified, enhanced plate image.
Lazy-loaded on first call.

Includes:
  - Indian number plate regex matching
  - Positional OCR confusion-char fixer (0↔O, 1↔I, etc.)
  - PaddleOCR result unpacking compatible with v2–v4 API

Output:
    {"text": "HR26AB1234", "confidence": 0.93, "raw": [...]}
"""
import gc
import re
import os
import cv2
import numpy as np
from typing import List

from app.config import OCR_LANGUAGE

# Indian plate regex: e.g. HR26AB1234 / MH02CY9999 / DL8C1234
_PLATE_RE = re.compile(
    r"[A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{1,4}", re.IGNORECASE
)

# Positional confusion map
_TO_ALPHA = {"0": "O", "1": "I", "5": "S", "8": "B", "2": "Z"}
_TO_DIGIT = {"O": "0", "I": "1", "S": "5", "B": "8",
             "Z": "2", "G": "6", "Q": "0", "D": "0"}
_ALPHA_SLOTS = set(range(0, 2)) | set(range(4, 7))
_DIGIT_SLOTS = set(range(2, 4)) | set(range(7, 12))


class PlateOCR:
    def __init__(self):
        self._ocr        = None
        self._ocr_tried  = False
        print(f"[PlateOCR] Initialized (lang={OCR_LANGUAGE})")

    # ------------------------------------------------------------------
    # Lazy load
    # ------------------------------------------------------------------
    def _load(self) -> bool:
        if self._ocr is not None:
            return True
        if self._ocr_tried:
            return False
        self._ocr_tried = True
        try:
            # Disable MKLDNN — causes ConvertPirAttribute2RuntimeAttribute crash on Windows CPUs
            os.environ["FLAGS_enable_pir_api"]  = "0"
            os.environ["FLAGS_use_onednn"]      = "0"

            from paddleocr import PaddleOCR
            self._ocr = PaddleOCR(
                use_angle_cls=False,
                lang=OCR_LANGUAGE[0] if len(OCR_LANGUAGE) == 1 else "en",
                ocr_version="PP-OCRv4",
                enable_mkldnn=False,
            )
            print("[PlateOCR] PaddleOCR loaded")
            return True
        except Exception as e:
            print(f"[PlateOCR] PaddleOCR load failed: {e}")
            return False

    # ------------------------------------------------------------------
    # Pre-processing
    # ------------------------------------------------------------------
    @staticmethod
    def _preprocess(img: np.ndarray) -> np.ndarray:
        """CLAHE → Otsu threshold → morphological close → back to BGR."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4)).apply(gray)
        _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        bw     = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
        return cv2.cvtColor(bw, cv2.COLOR_GRAY2BGR)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def read_text(self, plate_image: np.ndarray, conf_threshold: float = 0.4) -> dict:
        if plate_image is None or plate_image.size == 0:
            return {"text": "", "confidence": 0.0, "raw": []}

        preprocessed = self._preprocess(plate_image)
        gc.collect()

        if self._load():
            try:
                return self._run_paddle(preprocessed, conf_threshold)
            except Exception as e:
                print(f"[PlateOCR] Inference error: {e}")

        return {"text": "", "confidence": 0.0, "raw": []}

    # ------------------------------------------------------------------
    # PaddleOCR runner
    # ------------------------------------------------------------------
    def _run_paddle(self, img: np.ndarray, conf_threshold: float) -> dict:
        result = self._ocr.ocr(img)

        raw, texts, confs = [], [], []
        if not result or result[0] is None:
            return {"text": "", "confidence": 0.0, "raw": []}

        for line in result[0]:
            if not line:
                continue
            # Version-safe unpacking
            if isinstance(line, (list, tuple)) and len(line) == 2:
                _, text_res = line[0], line[1]
                if isinstance(text_res, (list, tuple)) and len(text_res) == 2:
                    text, conf = str(text_res[0]), float(text_res[1])
                else:
                    text, conf = str(text_res), 0.0
            else:
                text, conf = str(line), 0.0

            raw.append({"text": text, "confidence": round(conf, 4)})
            if conf >= conf_threshold and text.strip():
                texts.append(text.strip())
                confs.append(conf)

        best  = _best_candidate(texts)
        avg_c = float(np.mean(confs)) if confs else 0.0
        print(f"[PlateOCR] result: '{best}' (conf={avg_c:.2f})")
        return {"text": best, "confidence": round(avg_c, 4), "raw": raw}


# ------------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------------
def _clean(raw: str) -> str:
    if not raw:
        return ""
    text = raw.upper()
    text = re.sub(r"[^A-Z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text).strip()


def _fix_positions(s: str) -> str:
    chars = list(s)
    for i, ch in enumerate(chars):
        if i in _ALPHA_SLOTS and ch in _TO_ALPHA:
            chars[i] = _TO_ALPHA[ch]
        elif i in _DIGIT_SLOTS and ch in _TO_DIGIT:
            chars[i] = _TO_DIGIT[ch]
    return "".join(chars)


def _best_candidate(parts: List[str]) -> str:
    if not parts:
        return ""
    joined  = _clean(" ".join(parts))
    nospace = joined.replace(" ", "")
    for candidate in [joined, nospace, _fix_positions(nospace)]:
        m = _PLATE_RE.search(candidate)
        if m:
            return re.sub(r"\s+", "", m.group(0)).upper()
    return joined


# Singleton
plate_ocr = PlateOCR()
