"""
CCTV Face Reconstruction System
Entry point for the FastAPI application.
"""
import os

# Must be set BEFORE numpy/cv2/torch are imported
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import uvicorn
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    print("=" * 65)
    print(" CCTV Intelligence Suite v2.0")
    print("=" * 65)
    print(f" Frontend  : http://localhost:{port}")
    print(f" API Docs  : http://localhost:{port}/docs")
    print(f" Plate API : http://localhost:{port}/api/plates/detect")
    print("=" * 65)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )
