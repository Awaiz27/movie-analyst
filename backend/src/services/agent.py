"""
Agent factory and orchestration
Creates and manages LlamaIndex agents with Concentrate AI router
"""
import anyio
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import Memory
from src.services.concentrate_llm import ConcentrateResponsesLLM
from src.services.tool import (
    search_tool,
    details_tool,
    trending_tool,
    popular_tool,
    top_rated_tool,
    upcoming_tool,
    airing_today_tool,
    on_the_air_tool,
    similar_tool,
    recommendations_tool,
    discover_tool,
    find_id_tool
)
from src.core.config import settings
from src.core.logger import get_logger
from src.core.db import SessionLocal, ChatSession

logger = get_logger("agent")

# ── In-process cache: keep agents alive so Memory persists across messages ──
_agent_cache: dict[str, ReActAgent] = {}
_memory_cache: dict[str, Memory] = {}      # Memory objects keyed by session_id
_agent_model_cache: dict[str, str] = {}    # track which model each cached agent uses

def _ensure_session_exists(session_id: str) -> None:
    """Ensure a session row exists in the database (sync)."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(
                id=session_id,
                title=f"Cinematic Discussion {session_id[:8]}"
            )
            db.add(session)
            db.commit()
            logger.debug(f"Created session record: {session_id}")
    finally:
        db.close()


async def get_agent_for_session(session_id: str, model: str = None) -> tuple[ReActAgent, Memory]:
    """
    Get or create an agent for a session ID.
    Agents and their Memory objects are cached in-process so that
    conversation history persists across messages within the same session.
    
    Args:
        session_id: UUID session identifier
        model: Model to use - 'auto' (default), 'gpt-4', 'claude-3-sonnet', 'gemini-pro', etc.
        
    Returns:
        Tuple of (agent, memory) — the same Memory object is reused across calls
    """
    try:
        selected_model = model or settings.DEFAULT_MODEL

        # Return cached agent if session exists and model hasn't changed
        if session_id in _agent_cache:
            cached_model = _agent_model_cache.get(session_id)
            if cached_model == selected_model:
                logger.debug(f"Reusing cached agent for session: {session_id}")
                return _agent_cache[session_id], _memory_cache[session_id]
            else:
                logger.info(f"Model changed ({cached_model} → {selected_model}), recreating agent for {session_id}")
                del _agent_cache[session_id]
                del _agent_model_cache[session_id]
                _memory_cache.pop(session_id, None)

        logger.debug(f"Creating agent for session: {session_id}")
        
        # Ensure session exists in database without blocking the event loop
        await anyio.to_thread.run_sync(_ensure_session_exists, session_id)
        
        # Initialize LLM with Concentrate AI
        logger.info(f"Using model: {selected_model}")
        
        llm = ConcentrateResponsesLLM(
            model=selected_model,
            api_key=settings.CONCENTRATE_API_KEY,
            base_url=settings.CONCENTRATE_BASE_URL,
            timeout=settings.HTTP_TIMEOUT,
            default_tool_choice="none"
        )
        
        # Configure memory (local or remote) using the new Memory API
        memory_kwargs = {
            "session_id": session_id,
            "token_limit": settings.AGENT_MEMORY_TOKEN_LIMIT
        }
        if settings.MEMORY_DB_URI:
            memory_kwargs["async_database_uri"] = settings.MEMORY_DB_URI
        if settings.MEMORY_TABLE_NAME:
            memory_kwargs["table_name"] = settings.MEMORY_TABLE_NAME
        memory = Memory.from_defaults(**memory_kwargs)
        
        # Create agent worker with all available unified tools
        all_tools = [
            search_tool,                # Universal search (movies, TV, people)
            details_tool,               # Comprehensive details with credits, recommendations
            trending_tool,              # Trending content
            popular_tool,               # Popular movies/TV
            top_rated_tool,             # Top-rated movies/TV
            upcoming_tool,              # Upcoming releases
            airing_today_tool,          # TV airing today
            on_the_air_tool,            # On-the-air TV shows
            similar_tool,               # Similar content recommendations
            recommendations_tool,       # Personalized recommendations
            discover_tool,              # Advanced filtering & discovery
            find_id_tool                # Search by external IDs (IMDB, etc.)
        ]
        
        # Set max_output_tokens on LLM (not on agent – ReActAgent doesn't accept it)
        if settings.MAX_OUTPUT_TOKENS:
            llm.default_max_output_tokens = settings.MAX_OUTPUT_TOKENS

        agent = ReActAgent(
            tools=all_tools,
            llm=llm,
            system_prompt=settings.SYSTEM_PROMPT,
            verbose=settings.AGENT_VERBOSE,
        )
        
        # Cache the agent AND memory separately so memory persists across calls
        _agent_cache[session_id] = agent
        _memory_cache[session_id] = memory
        _agent_model_cache[session_id] = selected_model
        
        logger.debug(f"✓ Agent ready for session: {session_id} (cached)")
        
        return agent, memory
        
    except Exception as e:
        logger.error(f"Failed to create agent for session {session_id}: {e}")
        raise