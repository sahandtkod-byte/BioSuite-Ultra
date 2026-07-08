"""
BioSuite Ultra API Server — Entry Point

Run with:
    python -m biosuite.api.server
    # or
    uvicorn biosuite.api.server:app --host 0.0.0.0 --port 8000

Open API docs at: http://localhost:8000/docs
"""
import uvicorn

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("  BioSuite Ultra REST API Server")
    print("  100% Free • Open Source • No Paid Features")
    print("=" * 60)
    print()
    print("  API Documentation: http://localhost:8000/docs")
    print("  ReDoc Reference:   http://localhost:8000/redoc")
    print("  Health Check:      http://localhost:8000/health")
    print()
    print("  Press Ctrl+C to stop")
    print("=" * 60 + "\n")

    uvicorn.run(
        "biosuite.api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
