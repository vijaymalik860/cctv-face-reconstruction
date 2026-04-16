"""
AI Face Enhancement Service
Wraps GFPGAN + Real-ESRGAN for face restoration and background upscaling.
Runs on CPU (Intel UHD - no CUDA).

Memory-optimized: limits input size, uses gc.collect(), and provides
a lightweight OpenCV fallback for systems with limited virtual memory.
"""
import os
import sys
import gc
import cv2
import time
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List

from app.config import (
    MODEL_DIR, OUTPUT_DIR, DEVICE, 
    GFPGAN_MODEL_URL, REALESRGAN_MODEL_URL,
    FSRCNN_X2_URL, FSRCNN_X4_URL
)

# Max input dimension — prevents OOM on 16GB RAM with small page file
MAX_INPUT_DIM = 800


def _patch_basicsr_imports():
    """
    Patch basicsr to skip heavy dataset imports that pull in scipy.
    This avoids 'paging file too small' errors on low-memory systems.
    We stub out every function that downstream modules try to import
    from basicsr.data.degradations so the real module (which imports scipy)
    never gets loaded.
    """
    import types

    # Comprehensive dummy — covers ALL functions gfpgan / realesrgan reference
    dummy = types.ModuleType('basicsr.data.degradations')
    _stub_names = [
        'circular_lowpass_kernel', 'random_mixed_kernels',
        'random_add_gaussian_noise_pt', 'random_add_poisson_noise_pt',
        'random_add_gaussian_noise', 'random_add_poisson_noise',
        'generate_kernel', 'generate_sinc_kernel',
        'mesh_grid', 'sigma_matrix2', 'pdf2', 'cdf2',
        'bivariate_Gaussian', 'mass_center_shift',
    ]
    for name in _stub_names:
        setattr(dummy, name, lambda *a, **k: None)
    sys.modules['basicsr.data.degradations'] = dummy

    dummy_ds = types.ModuleType('basicsr.data.realesrgan_dataset')
    sys.modules['basicsr.data.realesrgan_dataset'] = dummy_ds

    dummy_paired = types.ModuleType('basicsr.data.realesrgan_paired_dataset')
    sys.modules['basicsr.data.realesrgan_paired_dataset'] = dummy_paired


class FaceEnhancer:
    """
    AI-powered face enhancement using GFPGAN and Real-ESRGAN.
    Models are lazy-loaded on first use to save memory.
    Falls back to OpenCV enhancement if AI models fail to load.
    """

    def __init__(self):
        self._gfpgan = None
        self._bg_upsampler = None
        self._device = DEVICE
        self._models_dir = Path(MODEL_DIR)
        self._models_dir.mkdir(parents=True, exist_ok=True)
        self._patched = False
        self._force_opencv = False
        print(f"[FaceEnhancer] Initialized (device: {DEVICE})")

    def _has_enough_memory(self, required_mb: int = 2000) -> bool:
        """Check if system has enough free virtual memory."""
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
            avail_mb = stat.ullAvailPhys / (1024 * 1024)
            print(f"[FaceEnhancer] Available RAM: {avail_mb:.0f} MB (need {required_mb} MB)")
            return avail_mb >= required_mb
        except Exception:
            return True  # If check fails, try anyway

    def _ensure_patched(self):
        """Apply basicsr patches before any model import."""
        if not self._patched:
            _patch_basicsr_imports()
            self._patched = True

    def _download_model(self, url: str, filename: str) -> str:
        """Download model weights if not already present."""
        model_path = self._models_dir / filename
        if model_path.exists():
            return str(model_path)

        print(f"[FaceEnhancer] Downloading model: {filename}...")
        import urllib.request
        urllib.request.urlretrieve(url, str(model_path))
        print(f"[FaceEnhancer] Model downloaded: {filename}")
        return str(model_path)

    def _get_bg_upsampler(self, upscale: int = 2):
        """Initialize Real-ESRGAN background upsampler."""
        if self._bg_upsampler is not None:
            return self._bg_upsampler

        self._ensure_patched()

        try:
            from basicsr.archs.rrdbnet_arch import RRDBNet
            from realesrgan import RealESRGANer

            if upscale == 2:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=2
                )
                model_url = REALESRGAN_MODEL_URL
                scale = 2
            else:
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64,
                    num_block=23, num_grow_ch=32, scale=4
                )
                model_url = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
                scale = 4

            model_path = self._download_model(
                model_url,
                f"RealESRGAN_x{scale}plus.pth"
            )

            self._bg_upsampler = RealESRGANer(
                scale=scale,
                model_path=model_path,
                model=model,
                tile=200,
                tile_pad=10,
                pre_pad=0,
                half=False  # CPU doesn't support half precision
            )
            print("[FaceEnhancer] Real-ESRGAN loaded")
            return self._bg_upsampler

        except Exception as e:
            print(f"[FaceEnhancer] Real-ESRGAN unavailable: {e}")
            return None

    def _load_gfpgan(self, upscale: int = 2):
        """Load GFPGAN model."""
        if self._gfpgan is not None:
            return self._gfpgan

        self._ensure_patched()

        try:
            from gfpgan import GFPGANer

            model_path = self._download_model(
                GFPGAN_MODEL_URL,
                "GFPGANv1.4.pth"
            )

            bg_upsampler = self._get_bg_upsampler(upscale)

            self._gfpgan = GFPGANer(
                model_path=model_path,
                upscale=upscale,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=bg_upsampler
            )
            print("[FaceEnhancer] GFPGAN loaded successfully")
            return self._gfpgan

        except Exception as e:
            print(f"[FaceEnhancer] GFPGAN load failed: {e}")
            raise RuntimeError(f"GFPGAN loading failed: {e}")

    # ------------------------------------------------------------------
    # Core resize helper
    # ------------------------------------------------------------------
    @staticmethod
    def _limit_size(image: np.ndarray, max_dim: int = MAX_INPUT_DIM) -> np.ndarray:
        """Down-scale so the largest side <= max_dim."""
        h, w = image.shape[:2]
        if max(h, w) <= max_dim:
            return image
        scale = max_dim / max(h, w)
        nw, nh = int(w * scale), int(h * scale)
        print(f"[FaceEnhancer] Resize {w}x{h} -> {nw}x{nh} (memory-safe)")
        resized = cv2.resize(image, (nw, nh), interpolation=cv2.INTER_AREA)
        gc.collect()
        return resized

    # ------------------------------------------------------------------
    # Public entry-point
    # ------------------------------------------------------------------
    def enhance_image(
        self,
        image: np.ndarray,
        model: str = "gfpgan",
        upscale: int = 2,
        face_enhance: bool = True,
        bg_enhance: bool = True,
        fidelity: float = 0.5
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray], float]:
        """
        Enhance an image using AI models with OpenCV fallback.

        Returns:
            (enhanced_image, cropped_faces, restored_faces, processing_time)
        """
        start_time = time.time()
        gc.collect()

        # Constrain input
        image = self._limit_size(image)

        if model == "fsrcnn":
            # FSRCNN is extremely lightweight, skip the 2GB memory check
            try:
                result = self._enhance_fsrcnn(image, upscale)
            except Exception as e:
                print(f"[FaceEnhancer] FSRCNN failed ({e}), using OpenCV fallback")
                result = self._enhance_opencv(image, upscale)
        else:
            # Check memory before trying heavy AI models (GFPGAN/RealESRGAN)
            use_ai = not self._force_opencv and self._has_enough_memory(2000)
            if use_ai:
                try:
                    if model == "gfpgan":
                        result = self._enhance_gfpgan(image, upscale, fidelity)
                    elif model == "realesrgan":
                        result = self._enhance_realesrgan_only(image, upscale)
                    else:
                        result = self._enhance_gfpgan(image, upscale, fidelity)
                except Exception as e:
                    print(f"[FaceEnhancer] Heavy AI failed ({e}), using FSRCNN fallback")
                    self._force_opencv = True  # Don't try heavy AI again this session
                    try:
                        result = self._enhance_fsrcnn(image, upscale)
                    except:
                        result = self._enhance_opencv(image, upscale)
            else:
                print("[FaceEnhancer] Low memory — using FSRCNN fallback directly")
                try:
                    result = self._enhance_fsrcnn(image, upscale)
                except:
                    result = self._enhance_opencv(image, upscale)

        gc.collect()
        return (*result, time.time() - start_time)

    # ------------------------------------------------------------------
    # FSRCNN path (Low RAM / Fast SR)
    # ------------------------------------------------------------------
    def _load_fsrcnn(self, upscale: int = 2):
        """Load OpenCV dnn_superres FSRCNN model."""
        import cv2
        from cv2 import dnn_superres

        model_name = "FSRCNN"
        if upscale == 2:
            model_url = FSRCNN_X2_URL
            filename = "FSRCNN_x2.pb"
            scale = 2
        else:
            model_url = FSRCNN_X4_URL
            filename = "FSRCNN_x4.pb"
            scale = 4

        model_path = self._download_model(model_url, filename)
        
        sr = dnn_superres.DnnSuperResImpl_create()
        sr.readModel(model_path)
        sr.setModel("fsrcnn", scale)
        return sr

    def _enhance_fsrcnn(self, image: np.ndarray, upscale: int = 2) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """Enhance using lightweight FSRCNN."""
        print(f"[FaceEnhancer] Enhancing with FSRCNN x{upscale} (Low RAM)...")
        sr = self._load_fsrcnn(upscale)
        
        # Super-resolve full background
        upscaled_bg = sr.upsample(image)
        
        # Currently not enhancing individual faces separately with FSRCNN as it's full-image
        # We just return the upscaled image. Face crops can be taken from it if needed later.
        
        print("[FaceEnhancer] FSRCNN done")
        return upscaled_bg, [], []

    # ------------------------------------------------------------------
    # GFPGAN path
    # ------------------------------------------------------------------
    def _enhance_gfpgan(self, image, upscale=2, fidelity=0.5):
        restorer = self._load_gfpgan(upscale)
        cropped, restored, out = restorer.enhance(
            image, has_aligned=False, only_center_face=False,
            paste_back=True, weight=fidelity
        )
        if out is None:
            return self._enhance_opencv(image, upscale)
        return out, cropped, restored

    # ------------------------------------------------------------------
    # Real-ESRGAN only path
    # ------------------------------------------------------------------
    def _enhance_realesrgan_only(self, image, upscale=2):
        up = self._get_bg_upsampler(upscale)
        if up is None:
            return self._enhance_opencv(image, upscale)
        try:
            out, _ = up.enhance(image, outscale=upscale)
            return out, [], []
        except Exception as e:
            print(f"[FaceEnhancer] Real-ESRGAN error: {e}")
            return self._enhance_opencv(image, upscale)

    # ------------------------------------------------------------------
    # OpenCV fallback  (lightweight, no scipy, no large allocations)
    # ------------------------------------------------------------------
    def _enhance_opencv(
        self, image: np.ndarray, upscale: int = 2
    ) -> Tuple[np.ndarray, List[np.ndarray], List[np.ndarray]]:
        """
        Lightweight OpenCV pipeline optimised for low-memory systems.
        Steps: upscale -> denoise -> contrast -> sharpen
        Every intermediate is released with gc.collect().
        """
        print("[FaceEnhancer] OpenCV enhancement (memory-safe)...")
        h, w = image.shape[:2]

        # 1. Upscale (bicubic)
        result = cv2.resize(image, (w * upscale, h * upscale),
                            interpolation=cv2.INTER_CUBIC)
        del image
        gc.collect()

        # 2. Light denoise — small Gaussian blend (< 1 MB overhead)
        blur = cv2.GaussianBlur(result, (3, 3), 0.5)
        result = cv2.addWeighted(result, 0.75, blur, 0.25, 0)
        del blur
        gc.collect()

        # 3. CLAHE on L channel only
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        del lab
        gc.collect()
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        L = clahe.apply(L)
        result = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
        del L, A, B
        gc.collect()

        # 4. Sharpen (unsharp mask)
        g = cv2.GaussianBlur(result, (0, 0), 2.0)
        result = cv2.addWeighted(result, 1.4, g, -0.4, 0)
        del g
        gc.collect()

        # 5. Very light bilateral smoothing (d=3 keeps memory small)
        result = cv2.bilateralFilter(result, d=3, sigmaColor=35, sigmaSpace=35)
        gc.collect()

        print(f"[FaceEnhancer] Done: {w}x{h} -> {w*upscale}x{h*upscale}")
        return result, [], []

    def enhance_plate(
        self,
        plate_crop: np.ndarray,
        upscale: int = 4,
        is_night: bool = False,
    ) -> np.ndarray:
        """
        Sharpen / upscale a single rectified number plate crop.

        Night mode (is_night=True) runs a dedicated dark-CCTV pipeline:
          1. Extreme CLAHE (clipLimit=8) to lift shadows
          2. Aggressive gamma correction (γ=0.35)
          3. Second CLAHE pass to restore local contrast
          4. Dark-channel dehazing
          5. Wiener-like deblur (Laplacian sharpening)
          6. Bilateral denoise
          7. Super-resolution (Real-ESRGAN or OpenCV bicubic)
          8. Final unsharp mask

        Day mode uses Real-ESRGAN → OpenCV fallback (existing pipeline).

        Args:
            plate_crop: BGR numpy array — rectified plate (e.g. 300×100)
            upscale:    upscale factor (default 4 for plates)
            is_night:   True if the source frame was detected as night/dark

        Returns:
            Enhanced BGR numpy array.
        """
        if plate_crop is None or plate_crop.size == 0:
            return plate_crop

        # Constrain input size
        plate_crop = self._limit_size(plate_crop, max_dim=600)

        # ------------------------------------------------------------------ #
        # NIGHT pipeline — dedicated dark CCTV processing                     #
        # ------------------------------------------------------------------ #
        if is_night:
            return self._enhance_plate_night(plate_crop, upscale)

        # ------------------------------------------------------------------ #
        # DAY pipeline — Real-ESRGAN → OpenCV fallback                        #
        # ------------------------------------------------------------------ #
        try:
            up = self._get_bg_upsampler(upscale)
            if up is not None:
                enhanced, _ = up.enhance(plate_crop, outscale=upscale)
                print(f"[FaceEnhancer] Plate enhanced with Real-ESRGAN ×{upscale}")
                return enhanced
        except Exception as e:
            print(f"[FaceEnhancer] Real-ESRGAN plate enhance failed: {e}")

        # Day fallback
        return self._enhance_plate_opencv(plate_crop, upscale,
                                          clahe_clip=3.0, gamma=None)

    def _enhance_plate_night(self, plate_crop: np.ndarray, upscale: int) -> np.ndarray:
        """
        Multi-stage enhancement for dark/night CCTV number plates.
        Designed for the exact scenario: parked car at night, dim plate,
        partial shadow, possible motion blur.
        """
        print("[FaceEnhancer] Night plate pipeline: CLAHE → dehaze → deblur → SR")
        img = plate_crop.copy()

        # ── Step 1: Extreme CLAHE on L channel ──────────────────────────────
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe1 = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(4, 4))
        L = clahe1.apply(L)
        img = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
        del lab, L, A, B
        gc.collect()

        # ── Step 2: Aggressive gamma lift (γ=0.35) ──────────────────────────
        lut = np.array(
            [min(255, int((i / 255.0) ** 0.35 * 255)) for i in range(256)],
            dtype=np.uint8
        )
        img = cv2.LUT(img, lut)

        # ── Step 3: Second CLAHE pass (restore local contrast after gamma) ───
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe2 = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(6, 6))
        L = clahe2.apply(L)
        img = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
        del lab, L, A, B
        gc.collect()

        # ── Step 4: Dark channel dehazing ───────────────────────────────────
        try:
            img_f = img.astype(np.float32) / 255.0
            dark  = np.min(img_f, axis=2)
            dark_blur = cv2.GaussianBlur(dark, (15, 15), 0)
            # Estimate atmospheric light (top-1% brightest pixels in dark channel)
            flat = dark_blur.flatten()
            atm_idx = np.argsort(flat)[-max(1, len(flat) // 100):]
            atm = np.max(img_f.reshape(-1, 3)[atm_idx], axis=0)
            atm = np.clip(atm, 0.7, 1.0)
            # Transmission map
            t = 1.0 - 0.85 * (dark_blur / (atm.max() + 1e-6))
            t = np.clip(t, 0.15, 1.0)
            t3 = np.stack([t, t, t], axis=2)
            dehazed = (img_f - atm) / t3 + atm
            img = np.clip(dehazed * 255, 0, 255).astype(np.uint8)
            del img_f, dark, dark_blur, t, t3, dehazed
            gc.collect()
        except Exception as e:
            print(f"[FaceEnhancer] Dehazing skipped: {e}")

        # ── Step 5: Wiener-style deblur (iterative Laplacian sharpening) ────
        # Approximate blind deblur via repeated unsharp mask
        for _ in range(2):
            blur = cv2.GaussianBlur(img, (0, 0), 1.2)
            img  = cv2.addWeighted(img, 1.6, blur, -0.6, 0)

        # ── Step 6: Bilateral denoise (preserves edges = character strokes) ─
        img = cv2.bilateralFilter(img, d=5, sigmaColor=40, sigmaSpace=40)
        gc.collect()

        # ── Step 7: Super-resolution ─────────────────────────────────────────
        # CRITICAL: We bypass Real-ESRGAN for night plates!
        # Real-ESRGAN hallucinates and heavily smooths faint/shadowed text (like '950'),
        # completely erasing it. We use FSRCNN or high-quality OpenCV interpolation instead.
        h, w = img.shape[:2]
        try:
            sr = self._load_fsrcnn(upscale)
            img = sr.upsample(img)
            print(f"[FaceEnhancer] Night plate SR: FSRCNN ×{upscale}")
        except Exception:
            img = cv2.resize(img, (w * upscale, h * upscale),
                             interpolation=cv2.INTER_CUBIC)
            print(f"[FaceEnhancer] Night plate SR: OpenCV Bicubic ×{upscale}")

        # ── Step 8: Final unsharp mask to crisp up characters ────────────────
        blur_final = cv2.GaussianBlur(img, (0, 0), 1.0)
        img = cv2.addWeighted(img, 1.4, blur_final, -0.4, 0)

        print(f"[FaceEnhancer] Night plate done: {w}×{h} → {img.shape[1]}×{img.shape[0]}")
        gc.collect()
        return img

    def _enhance_plate_opencv(self, plate_crop: np.ndarray, upscale: int,
                               clahe_clip: float = 3.0,
                               gamma: float = None) -> np.ndarray:
        """Lightweight OpenCV plate enhancement (day fallback)."""
        print("[FaceEnhancer] Using OpenCV fallback for plate enhancement")
        h, w = plate_crop.shape[:2]
        result = cv2.resize(plate_crop, (w * upscale, h * upscale),
                            interpolation=cv2.INTER_CUBIC)
        # CLAHE on L channel
        lab = cv2.cvtColor(result, cv2.COLOR_BGR2LAB)
        L, A, B = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(4, 4))
        L = clahe.apply(L)
        result = cv2.cvtColor(cv2.merge([L, A, B]), cv2.COLOR_LAB2BGR)
        del lab, L, A, B
        if gamma is not None:
            lut = np.array(
                [min(255, int((i / 255.0) ** gamma * 255)) for i in range(256)],
                dtype=np.uint8)
            result = cv2.LUT(result, lut)
        # Unsharp mask
        blur   = cv2.GaussianBlur(result, (0, 0), 1.5)
        result = cv2.addWeighted(result, 1.5, blur, -0.5, 0)
        gc.collect()
        return result

    def save_results(
        self,
        job_id: str,
        enhanced_img: np.ndarray,
        cropped_faces: List[np.ndarray],
        restored_faces: List[np.ndarray]
    ) -> dict:
        """Save enhanced image and face crops to disk."""
        results = {"enhanced_path": "", "face_paths": []}

        # Save enhanced full image
        output_path = OUTPUT_DIR / f"{job_id}_enhanced.jpg"
        cv2.imwrite(str(output_path), enhanced_img)
        results["enhanced_path"] = str(output_path)

        # Save face crops
        face_dir = OUTPUT_DIR / "faces"
        face_dir.mkdir(parents=True, exist_ok=True)

        for i, (cropped, restored) in enumerate(
            zip(cropped_faces or [], restored_faces or [])
        ):
            crop_path = face_dir / f"{job_id}_face_{i}_cropped.jpg"
            cv2.imwrite(str(crop_path), cropped)

            enhanced_face_path = face_dir / f"{job_id}_face_{i}_enhanced.jpg"
            cv2.imwrite(str(enhanced_face_path), restored)

            results["face_paths"].append({
                "cropped": str(crop_path),
                "enhanced": str(enhanced_face_path)
            })

        return results

    def get_model_info(self) -> dict:
        """Get information about loaded models."""
        return {
            "device": DEVICE,
            "gfpgan_loaded": self._gfpgan is not None,
            "bg_upsampler_loaded": self._bg_upsampler is not None,
            "models_dir": str(self._models_dir),
            "available_models": ["gfpgan", "realesrgan", "fsrcnn", "opencv"]
        }


# Singleton instance
face_enhancer = FaceEnhancer()
