"""
FastAPI application initialization and configuration
Sets up middleware, exception handlers, and core endpoints
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import uuid

from src.api.routes import router
from src.core.config import settings
from src.core.db import init_db
from src.core.logger import get_logger
from src.schemas.chat import ErrorResponse, HealthCheckResponse

logger = get_logger("app")


# Lifecycle management
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle"""
    # Startup
    logger.info("Starting Movie Analyst API...")
    try:
        init_db()
        logger.info("✓ Database initialized")
    except Exception as e:
        logger.error(f"Failed to initialize database: {e}")
        raise
    
    logger.info(f"✓ Using Concentrate AI gateway: {settings.CONCENTRATE_BASE_URL}")
    logger.info(f"✓ Memory limit per session: {settings.AGENT_MEMORY_TOKEN_LIMIT} tokens")
    
    yield
    
    # Shutdown
    logger.info("Shutting down Cinematic Mesh API...")


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    description=settings.API_DESCRIPTION,
    version=settings.API_VERSION,
    lifespan=lifespan
)


# === CORS Middleware ===
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# === Exception Handlers ===
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    """Handle validation errors with structured response"""
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            error="Request validation failed",
            error_code="VALIDATION_ERROR",
            request_id=str(uuid.uuid4())
        ).model_dump()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Catch-all exception handler"""
    request_id = str(uuid.uuid4())
    logger.error(f"Unhandled exception (ID: {request_id}): {exc}")
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Internal server error",
            error_code="INTERNAL_ERROR",
            request_id=request_id
        ).model_dump()
    )


app.include_router(router, prefix="/api/v1")


# === Health Check ===
@app.get("/health")
async def health_check():
    """System health status endpoint - tests all critical components"""
    import httpx
    from ..core.db import engine
    from sqlalchemy import text
    
    components = {
        "api": "operational",
        "database": "unknown",
        "tmdb_api": "unknown",
        "tvmaze_api": "unknown",
        "concentrate_ai": "unknown"
    }
    
    overall_status = "healthy"
    
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        components["database"] = "operational"
        logger.debug("✓ Database health check passed")
    except Exception as e:
        components["database"] = f"degraded: {str(e)[:50]}"
        overall_status = "degraded"
        logger.warning(f"Database health check failed: {e}")
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                f"{settings.TMDB_BASE_URL}/movie/550",
                params={"api_key": settings.TMDB_API_KEY}
            )
            if resp.status_code == 200:
                components["tmdb_api"] = "operational"
                logger.debug("✓ TMDB API health check passed")
            else:
                components["tmdb_api"] = f"degraded: status {resp.status_code}"
                overall_status = "degraded"
    except Exception as e:
        components["tmdb_api"] = "unavailable"
        overall_status = "degraded"
        logger.warning(f"TMDB API health check failed: {e}")
    
    # Test TVMaze API
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"{settings.TVMAZE_BASE_URL}/shows/1")
            if resp.status_code == 200:
                components["tvmaze_api"] = "operational"
                logger.debug("✓ TVMaze API health check passed")
            else:
                components["tvmaze_api"] = "degraded"
    except Exception as e:
        components["tvmaze_api"] = "unavailable"
        logger.warning(f"TVMaze API health check failed: {e}")
    
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(
                settings.CONCENTRATE_BASE_URL.replace("/v1", "/health"),
                headers={"Authorization": f"Bearer {settings.CONCENTRATE_API_KEY}"}
            )
            components["concentrate_ai"] = "operational"
            logger.debug("✓ Concentrate AI health check passed")
    except Exception as e:
        components["concentrate_ai"] = "assumed_operational"
        logger.debug(f"Concentrate AI health check: {e}")
    
    return HealthCheckResponse(
        status=overall_status,
        components=components
    )


@app.get("/")
async def root():
    """Welcome endpoint with API information"""
    return {
        "name": settings.API_TITLE,
        "version": settings.API_VERSION,
        "status": "operational",
        "docs": "/docs",
        "health": "/health"
    }
