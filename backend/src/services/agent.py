"""
Agent factory and orchestration.

- Every user + assistant message is persisted to SQLite (ChatMessage table).
- Fresh Memory is hydrated from DB on each request
- Agents created on-the-fly per request
  always completes and is saved to the database.
"""
import asyncio
import uuid as _uuid
from functools import partial

import anyio
from llama_index.core.agent import ReActAgent
from llama_index.core.memory import Memory
from llama_index.core.llms import ChatMessage as LlamaChatMessage

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
from src.core.db import SessionLocal, ChatSession, ChatMessage

logger = get_logger("agent")

# ── Background tasks: session_id -> list[asyncio.Task] ──
_background_tasks: dict[str, list[asyncio.Task]] = {}

# ── Shared tool instances (stateless, safe to reuse) ──
_ALL_TOOLS = [
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
    find_id_tool,
]


# ---------------------------------------------------------------------------
# Synchronous DB helpers (called via anyio.to_thread.run_sync)
# ---------------------------------------------------------------------------

def _ensure_session_exists(session_id: str) -> None:
    """Ensure a session row exists in the database."""
    db = SessionLocal()
    try:
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            session = ChatSession(
                id=session_id,
                title=f"New Chat {session_id[:8]}"
            )
            db.add(session)
            db.commit()
            logger.debug(f"Created session record: {session_id}")
    finally:
        db.close()


def _load_history(session_id: str) -> list[dict]:
    """
    Load all messages for a session ordered by creation time.
    Returns a list of {role, content} dicts.
    """
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        history = [{"role": r.role, "content": r.content} for r in rows]
        logger.debug(f"Loaded {len(history)} messages for session {session_id}")
        return history
    finally:
        db.close()


def _save_messages(session_id: str, messages: list[dict]) -> None:
    """
    Persist a batch of messages (user + assistant) atomically.
    Each dict must have 'role' and 'content' keys.
    """
    db = SessionLocal()
    try:
        for msg in messages:
            db.add(ChatMessage(
                id=str(_uuid.uuid4()),
                session_id=session_id,
                role=msg["role"],
                content=msg["content"],
            ))
        db.commit()
        logger.debug(f"Saved {len(messages)} messages for session {session_id}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Memory builder — hydrate from DB each time (no cache)
# ---------------------------------------------------------------------------

async def _build_memory(session_id: str) -> Memory:
    """
    Create a fresh Memory and populate it with all previous messages
    from the database.  No in-memory caching — every call reads from
    the DB so we always have the latest state and can scale horizontally.
    """
    await anyio.to_thread.run_sync(partial(_ensure_session_exists, session_id))
    history = await anyio.to_thread.run_sync(partial(_load_history, session_id))

    memory = Memory.from_defaults(
        session_id=session_id,
        token_limit=settings.AGENT_MEMORY_TOKEN_LIMIT,
    )

    # Hydrate memory with previous conversation
    for msg in history:
        memory.put(LlamaChatMessage(role=msg["role"], content=msg["content"]))

    logger.debug(
        f"Built memory for session {session_id} "
        f"with {len(history)} historical messages"
    )
    return memory


# ---------------------------------------------------------------------------
# Agent factory (created per-request — cheap)
# ---------------------------------------------------------------------------

def create_agent(model: str | None = None) -> ReActAgent:
    """
    Create a fresh ReActAgent.  Agents are lightweight wrappers around an
    LLM + tool list, so there is negligible overhead in creating one per
    request.
    """
    selected_model = model or settings.DEFAULT_MODEL
    logger.debug(f"Creating agent with model: {selected_model}")

    llm = ConcentrateResponsesLLM(
        model=selected_model,
        api_key=settings.CONCENTRATE_API_KEY,
        base_url=settings.CONCENTRATE_BASE_URL,
        timeout=settings.HTTP_TIMEOUT,
        default_tool_choice="none",
    )

    if settings.MAX_OUTPUT_TOKENS:
        llm.default_max_output_tokens = settings.MAX_OUTPUT_TOKENS

    return ReActAgent(
        tools=_ALL_TOOLS,
        llm=llm,
        system_prompt=settings.SYSTEM_PROMPT,
        verbose=settings.AGENT_VERBOSE,
    )


# ---------------------------------------------------------------------------
# Background task execution
# ---------------------------------------------------------------------------

def _cleanup_task(session_id: str, task: asyncio.Task) -> None:
    """Remove *task* from the tracking list for *session_id*."""
    tasks = _background_tasks.get(session_id, [])
    try:
        tasks.remove(task)
    except ValueError:
        pass
    if not tasks:
        _background_tasks.pop(session_id, None)


async def _run_agent_task(
    session_id: str,
    agent: ReActAgent,
    message: str,
    memory: Memory,
) -> str:
    """
    Execute the agent and persist the assistant response to the database.

    The coroutine runs inside an ``asyncio.Task`` so it survives client
    disconnects.  The *user* message was already saved to the DB in
    ``submit_agent_task``, so here we only save the assistant reply.
    """
    try:
        handler = agent.run(message, memory=memory, max_iterations=100)
        response = await handler
        result = str(response).strip()

        # Persist only the assistant reply (user message already in DB)
        await anyio.to_thread.run_sync(partial(
            _save_messages,
            session_id,
            [{"role": "assistant", "content": result}],
        ))

        logger.info(
            f"Background task completed for session {session_id} "
            f"({len(result)} chars)"
        )
        return result

    except asyncio.CancelledError:
        logger.warning(f"Background task cancelled for session {session_id}")
        raise
    except Exception as e:
        logger.error(
            f"Background task failed for session {session_id}: {e}",
            exc_info=True,
        )
        raise


async def submit_agent_task(
    session_id: str,
    message: str,
    model: str | None = None,
) -> asyncio.Task:
    """
    Submit an agent run as a background ``asyncio.Task``.

    1. Hydrate a fresh Memory from the DB (no cache).
    2. **Save the user message to the DB immediately** — this ensures
       the prompt is persisted even if the agent task crashes or the
       client disconnects.
    3. Create a fresh agent (cheap).
    4. Launch the task — it survives client disconnects and saves
       only the *assistant* response to the database on completion.

    Returns the ``asyncio.Task`` so callers can await / stream from it.
    """
    memory = await _build_memory(session_id)

    # ── Persist user message NOW (before the agent runs) ──
    await anyio.to_thread.run_sync(partial(
        _save_messages,
        session_id,
        [{"role": "user", "content": message}],
    ))
    # Note: agent.run() will add the user message to Memory internally,
    # so we do NOT call memory.put() here (that would create a duplicate).

    agent = create_agent(model)

    task = asyncio.create_task(
        _run_agent_task(session_id, agent, message, memory),
        name=f"agent-{session_id}",
    )
    task.add_done_callback(lambda t: _cleanup_task(session_id, t))

    _background_tasks.setdefault(session_id, []).append(task)
    logger.info(f"Submitted agent task for session {session_id}")
    return task


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_active_tasks(session_id: str) -> list[asyncio.Task]:
    """Return all currently running tasks for *session_id*."""
    return [t for t in _background_tasks.get(session_id, []) if not t.done()]


def drop_session(session_id: str) -> None:
    """Cancel running tasks for *session_id*.  No cache to clear."""
    tasks = _background_tasks.pop(session_id, [])
    for task in tasks:
        if not task.done():
            task.cancel()
            logger.info(f"Cancelled running task for deleted session {session_id}")


def cancel_session_tasks(session_id: str) -> int:
    """Cancel all running tasks for *session_id* without removing session data.

    Returns the number of tasks that were cancelled.  Used by the frontend
    when a timeout is reached.
    """
    cancelled = 0
    for task in _background_tasks.get(session_id, []):
        if not task.done():
            task.cancel()
            cancelled += 1
            logger.info(f"Timeout-cancelled task for session {session_id}")
    return cancelled