"""
Application Configuration
Manages all settings, paths, and environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Directory paths
UPLOAD_DIR = BASE_DIR / os.getenv("UPLOAD_DIR", "uploads")
OUTPUT_DIR = BASE_DIR / os.getenv("OUTPUT_DIR", "outputs")
MODEL_DIR = BASE_DIR / os.getenv("MODEL_DIR", "models")
STATIC_DIR = BASE_DIR / "static"

# Create directories if they don't exist
for directory in [UPLOAD_DIR, OUTPUT_DIR, MODEL_DIR,
                  OUTPUT_DIR / "faces", OUTPUT_DIR / "videos",
                  OUTPUT_DIR / "frames", UPLOAD_DIR / "videos"]:
    directory.mkdir(parents=True, exist_ok=True)

# Database
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/cctv_face_db"
)

# Server
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))

# Model settings
DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "gfpgan")
DEFAULT_UPSCALE = int(os.getenv("DEFAULT_UPSCALE", "2"))

# Device (Intel UHD - CPU only, no CUDA)
DEVICE = "cpu"

# Supported file types
ALLOWED_IMAGE_TYPES = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff"}
ALLOWED_VIDEO_TYPES = {".mp4", ".avi", ".mov", ".mkv", ".wmv"}
MAX_FILE_SIZE = 100 * 1024 * 1024  # 100MB

# GFPGAN / Real-ESRGAN / FSRCNN model URLs
GFPGAN_MODEL_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"
REALESRGAN_MODEL_URL = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth"
FSRCNN_X2_URL = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x2.pb"
FSRCNN_X4_URL = "https://github.com/Saafke/FSRCNN_Tensorflow/raw/master/models/FSRCNN_x4.pb"
