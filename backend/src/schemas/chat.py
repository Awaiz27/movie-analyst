"""
Pydantic models for chat API endpoints
Provides type-safe request/response validation
"""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime


class ChatRequest(BaseModel):
    """Incoming chat message request"""
    session_id: str = Field(..., min_length=1, description="UUID session identifier")
    message: str = Field(..., min_length=1, max_length=2000, description="User message")
    model: Optional[str] = Field(None, description="Model to use: 'auto', 'gpt-4', 'claude-3-sonnet', 'gemini-pro', etc.")
    stream: bool = Field(default=False, description="Enable streaming response")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "message": "Tell me about Inception"
            }
        }


class ChatResponse(BaseModel):
    """Response from agent processing"""
    content: str = Field(..., description="Agent's response")
    session_id: str = Field(..., description="Session reference")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": "Inception is a 2010 science fiction film...",
                "session_id": "550e8400-e29b-41d4-a716-446655440000",
                "timestamp": "2026-02-07T10:30:00Z"
            }
        }


class SessionCreateResponse(BaseModel):
    """Response for new session creation"""
    session_id: str = Field(..., description="New session UUID")
    created_at: datetime = Field(default_factory=datetime.utcnow)


class ErrorResponse(BaseModel):
    """Standardized error response"""
    error: str = Field(..., description="Error message")
    error_code: str = Field(..., description="Error code for client handling")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    request_id: Optional[str] = Field(None, description="Trace ID for debugging")


class MovieMetadataResponse(BaseModel):
    """Movie metadata from TMDB"""
    title: str
    release_date: Optional[str]
    overview: str
    rating: float = Field(..., alias="vote_average")
    budget: Optional[int]
    revenue: Optional[int]
    runtime: int
    genres: List[Dict[str, Any]]


class TVShowResponse(BaseModel):
    """TV show data from TVMaze"""
    name: str
    status: str
    premiere_date: Optional[str] = Field(None, alias="premiered")
    network: Optional[str]
    next_episode: Optional[Dict[str, Any]]
    summary: Optional[str]


class HealthCheckResponse(BaseModel):
    """System health status"""
    status: str = Field(..., description="System status: 'healthy' or 'degraded'")
    components: Dict[str, str] = Field(..., description="Individual component statuses")
    timestamp: datetime = Field(default_factory=datetime.utcnow)
