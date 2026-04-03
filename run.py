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
    
    print("=" * 60)
    print("🔬 CCTV Face Reconstruction System")
    print("=" * 60)
    print(f"🌐 Server: http://localhost:{port}")
    print(f"📡 API Docs: http://localhost:{port}/docs")
    print("=" * 60)
    
    uvicorn.run(
        "app.main:app",
        host=host,
        port=port,
        reload=True
    )
