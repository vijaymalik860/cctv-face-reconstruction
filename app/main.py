"""
CCTV Face Reconstruction System
FastAPI Main Application
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from app.config import UPLOAD_DIR, OUTPUT_DIR, STATIC_DIR
from app.database import init_db
from app.routes.enhance import router as enhance_router

# ============================================================
# App Creation
# ============================================================

app = FastAPI(
    title="🔬 CCTV Face Reconstruction System",
    description="AI-powered face restoration from low-quality CCTV footage using GFPGAN & Real-ESRGAN",
    version="1.0.0",
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

# Serve uploaded files
app.mount("/files/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")

# Serve output files
app.mount("/files/outputs", StaticFiles(directory=str(OUTPUT_DIR)), name="outputs")

# Serve static frontend files (CSS, JS)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ============================================================
# Routes
# ============================================================

app.include_router(enhance_router)


@app.get("/", include_in_schema=False)
async def serve_frontend():
    """Serve the main HTML page."""
    return FileResponse(str(STATIC_DIR / "index.html"))


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "CCTV Face Reconstruction System",
        "version": "1.0.0"
    }


# ============================================================
# Startup & Shutdown Events
# ============================================================

@app.on_event("startup")
async def startup_event():
    """Initialize database and load models on startup."""
    print("\n" + "=" * 60)
    print("🔬 CCTV Face Reconstruction System - Starting Up...")
    print("=" * 60)

    # Initialize database tables
    try:
        init_db()
        print("✅ Database initialized")
    except Exception as e:
        print(f"⚠️  Database initialization warning: {e}")
        print("   Make sure PostgreSQL is running and database exists!")

    print("=" * 60)
    print("🌐 Frontend: http://localhost:8000")
    print("📡 API Docs: http://localhost:8000/docs")
    print("=" * 60 + "\n")


@app.on_event("shutdown")
async def shutdown_event():
    print("\n🔴 CCTV Face Reconstruction System - Shutting down...")
