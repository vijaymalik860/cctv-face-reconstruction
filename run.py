"""
CCTV Face Reconstruction System
Entry point for the FastAPI application.
"""
import uvicorn
from dotenv import load_dotenv
import os

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
