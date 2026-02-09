"""
Application entry point
Runs the FastAPI server with uvicorn
"""
from src.api.main_app import app

if __name__ == "__main__":
    import uvicorn
    
    # This is run via Dockerfile CMD or local development
    # Configuration via environment variables and .env file
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )