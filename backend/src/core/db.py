"""
Database initialization and session management
Handles SQLite database for chat history persistence
Auto-creates database and tables on first run
"""
from sqlalchemy import create_engine, Column, String, DateTime, JSON, event, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.engine import Engine
import datetime
import os
from pathlib import Path

from src.core.config import settings
from src.core.logger import get_logger

logger = get_logger("database")

Base = declarative_base()


class ChatSession(Base):
    """Represents a user chat session"""
    __tablename__ = "chat_sessions"
    
    id = Column(String, primary_key=True)  # UUID
    title = Column(String)  # e.g., "Review of Inception"
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc))
    updated_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc), onupdate=datetime.datetime.now(datetime.timezone.utc))
    metadata_info = Column(JSON)  # For future extensibility


class ChatMessage(Base):
    """Represents individual messages in a session"""
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True)
    session_id = Column(String, index=True)
    role = Column(String)  # "user" or "assistant"
    content = Column(String)
    created_at = Column(DateTime, default=datetime.datetime.now(datetime.timezone.utc), index=True)
    


# Create database engine
def get_database_url():
    """Construct database URL from settings and ensure directory exists"""
    db_path = settings.DATABASE_PATH
    db_dir = os.path.dirname(db_path)
    
    # Create directory if it doesn't exist (works for both /app/data and ./data)
    if db_dir:
        Path(db_dir).mkdir(parents=True, exist_ok=True)
        logger.debug(f"Ensured database directory exists: {db_dir}")
    
    return f"sqlite:///{db_path}"


engine = create_engine(
    get_database_url(),
    connect_args={"check_same_thread": False},
    pool_size=settings.DATABASE_POOL_SIZE
)


@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_conn, connection_record):
    """Optimize SQLite for concurrent access"""
    cursor = dbapi_conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")  # Write-Ahead Logging
    cursor.execute("PRAGMA synchronous=NORMAL")  # Performance optimization
    cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def init_db():
    """Initialize database tables - runs automatically on app startup"""
    try:
        db_path = settings.DATABASE_PATH
        logger.info(f"Initializing database at: {db_path}")
        
        # Create all tables (idempotent - safe to call multiple times)
        Base.metadata.create_all(bind=engine)
        
        # Verify database is accessible
        with engine.connect() as connection:
            connection.execute(text("SELECT 1"))
        
        logger.info("✓ Database initialized and verified - tables ready for use")
        return True
        
    except Exception as e:
        logger.error(f"✗ Failed to initialize database: {e}")
        raise SystemExit(f"Database initialization failed: {e}")


def get_db_session():
    """Dependency to get database session"""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()