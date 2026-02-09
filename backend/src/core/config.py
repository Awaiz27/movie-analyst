import os
import logging
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

logger = logging.getLogger(__name__)


class Settings(BaseSettings):
    """
    Application settings
    """

    CONCENTRATE_API_KEY: str = Field(
        min_length=1,
        description="API key for Concentrate AI gateway"
    )
    TMDB_API_KEY: str = Field(
        min_length=1,
        description="API key for The Movie Database (TMDB)"
    )
    
    CONCENTRATE_BASE_URL: str = Field(
        default="https://api.concentrate.ai/v1",
        description="Concentrate AI API endpoint"
    )
    TMDB_BASE_URL: str = Field(
        default="https://api.themoviedb.org/3",
        description="TMDB API base URL"
    )
    TVMAZE_BASE_URL: str = Field(
        default="https://api.tvmaze.com",
        description="TVMaze API base URL"
    )
    
    # === DATABASE CONFIGURATION ===
    DATABASE_PATH: str = Field(
        default="./chat_history.db",
        description="SQLite database location"
    )
    DATABASE_POOL_SIZE: int = Field(default=5, ge=1, le=20)
    
    # === LOGGING CONFIGURATION ===
    LOG_LEVEL: str = Field(default="INFO")
    
    # === MODEL CONFIGURATION ===
    DEFAULT_MODEL: str = Field(
        default="gpt-4o-mini",
        description="Model to use: 'auto' for Concentrate routing, or specific model like 'gpt-4', 'claude-3-sonnet', 'gemini-pro'"
    )
    
    # === API CONFIGURATION ===
    API_TITLE: str = Field(default="MIVIE ANALYST API")
    API_VERSION: str = Field(default="1.0.0")
    API_DESCRIPTION: str = Field(
        default="Action-Oriented Research Agent for Cinematic Insights"
    )
    
    # === AGENT CONFIGURATION ===
    AGENT_MEMORY_TOKEN_LIMIT: int = Field(default=4000, ge=1000)
    AGENT_VERBOSE: bool = Field(default=False)
    SYSTEM_PROMPT: str = Field(
        default=(
            "You are Movie Analyst, an expert cinematic research assistant. "
            "You have access to live TMDB tools to search movies, TV shows, trending content, "
            "top-rated media, upcoming releases, and more. "
            "ALWAYS use your available tools to fetch real data — never make up or hallucinate results. "
            "Present data clearly with titles, ratings, release dates, and key facts. "
            # "Be concise and accurate."
        ),
        description="System prompt for the agent"
    )
    MAX_OUTPUT_TOKENS: int = Field(default=1000, ge=100, le=5000)
    MEMORY_DB_URI: str | None = Field(
        default=None,
        description="Optional async database URI for remote agent memory"
    )
    MEMORY_TABLE_NAME: str | None = Field(
        default=None,
        description="Optional table name for Memory storage"
    )
    
    # === RETRY & TIMEOUT CONFIGURATION ===
    HTTP_TIMEOUT: int = Field(default=60, ge=5, le=300)
    HTTP_RETRIES: int = Field(default=3, ge=1, le=10)
    RETRY_BACKOFF_FACTOR: float = Field(default=1.0, ge=0.1)
    
    # === SECURITY ===
    ALLOWED_ORIGINS: list[str] | str = Field(
        default=["http://localhost:8501", "http://frontend:8501"],
        description="CORS allowed origins"
    )
    
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=True,
        env_delimiter=",")

    @field_validator("CONCENTRATE_API_KEY", "TMDB_API_KEY", mode="after")
    @classmethod
    def validate_api_keys(cls, v: str) -> str:
        """Ensure API keys are not empty or whitespace-only"""
        if not v or not v.strip():
            raise ValueError("API Key cannot be empty or whitespace-only")
        return v.strip()
    
    @field_validator("LOG_LEVEL", mode="after")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        """Ensure valid log level"""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"LOG_LEVEL must be one of {valid_levels}")
        return v.upper()

    @field_validator("ALLOWED_ORIGINS", mode="before")
    @classmethod
    def normalize_allowed_origins(cls, v):
        """Allow comma-separated env var values for CORS origins."""
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v


def get_settings() -> Settings:
    """Factory function to load and validate settings"""
    try:
        settings = Settings()
        logger.info("✓ Configuration loaded and validated successfully")
        logger.debug(f"Database: {settings.DATABASE_PATH}")
        logger.debug(f"Log Level: {settings.LOG_LEVEL}")
        return settings
    except Exception as e:
        logger.critical(f"FATAL: Configuration validation failed: {e}")
        raise SystemExit(f"Configuration Error: {e}")


try:
    settings = get_settings()
except SystemExit:
    raise