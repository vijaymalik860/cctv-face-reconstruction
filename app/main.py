"""
CCTV Intelligence Suite v2.0
FastAPI Main Application
"""
import sys
import io
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Force UTF-8 output on Windows to avoid emoji encoding errors
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

from app.config import UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR
from app.database import init_db
from app.routes.enhance import router as enhance_router
from app.routes.plates  import router as plates_router

# ============================================================
# App Creation
# ============================================================

app = FastAPI(
    title="CCTV Intelligence Suite",
    description=(
        "AI-powered face restoration & vehicle number plate detection "
        "from low-quality CCTV footage. "
        "Uses GFPGAN, Real-ESRGAN, YOLOv8, WPOD-NET, and PaddleOCR."
    ),
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# ============================================================
# CORS Middleware
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============================================================
# Static Files & File Serving
# ============================================================

# Serve uploaded files (faces)
app.mount("/files/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Serve output files (faces + frames)
app.mount("/files/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve plate output files (per-job subdirectories)
app.mount("/files/plates", StaticFiles(directory=str(OUTPUT_DIR / "plates")), name="plate_outputs")

# Serve static frontend files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================
# Routes
# ============================================================

app.include_router(enhance_router)
app.include_router(plates_router)


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CCTV Intelligence Suite",
        "version": "2.0.0"
    }


# ============================================================
# Startup & Shutdown Events
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database and load models on startup."""
    sep = "=" * 65
    print(f"\n{sep}")
    print(" CCTV Intelligence Suite v2.0 - Starting Up...")
    print(sep)

    try:
        init_db()
        print("[OK] Database initialized (face + plate tables)")
    except Exception as e:
        print(f"[WARN] Database warning: {e}")
        print("   Make sure PostgreSQL is running and database exists!")

    print(sep)
    print(" Frontend  : http://localhost:8000")
    print(" API Docs  : http://localhost:8000/docs")
    print(" Face API  : POST /api/enhance/image")
    print(" Plate API : POST /api/plates/detect")
    print(f"{sep}\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n[STOP] CCTV Intelligence Suite - Shutting down...")
