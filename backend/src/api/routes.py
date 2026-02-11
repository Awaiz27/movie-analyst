import asyncio
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import uuid
import json
from typing import Optional, AsyncGenerator

from src.services.agent import submit_agent_task, drop_session, get_active_tasks, cancel_session_tasks
from src.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SessionCreateResponse,
    ErrorResponse
)
from src.core.logger import get_logger
from src.core.db import SessionLocal, ChatMessage, ChatSession
from src.services.tool import NotFoundError, APIError


logger = get_logger("routes")
router = APIRouter()


async def stream_chat_response(
    task: asyncio.Task,
    session_id: str,
    request_id: str,
) -> AsyncGenerator[str, None]:
    """Stream agent response as SSE events.

    The actual agent work runs inside *task* (a background ``asyncio.Task``).
    ``asyncio.shield`` ensures the task keeps running even if this generator
    is cancelled (e.g. the user switches to another chat and the SSE
    connection is dropped).  The agent will finish, and the response will
    be written into session Memory automatically.
    """
    try:
        logger.info(f"[{request_id}] Starting streaming response")

        # shield() prevents the task from being cancelled when FastAPI
        # cancels this generator on client disconnect.
        final_text = await asyncio.shield(task)

        logger.debug(
            f"[{request_id}] Final response ({len(final_text)} chars): "
            f"{final_text[:120]}..."
        )

        if final_text:
            # Stream word-by-word for a natural typing feel
            words = final_text.split(' ')
            chunk_size = 3
            for i in range(0, len(words), chunk_size):
                batch = words[i:i + chunk_size]
                chunk = ' '.join(batch) + (' ' if i + chunk_size < len(words) else '')
                yield f"data: {json.dumps({'content': chunk, 'session_id': session_id})}\n\n"
                await asyncio.sleep(0.03)

        yield f"data: {json.dumps({'session_id': session_id, 'done': True})}\n\n"
        logger.info(f"[{request_id}] Streaming completed")

    except asyncio.CancelledError:
        # Client disconnected — the shielded task continues in the background
        # and will write its response into Memory when done.
        logger.info(
            f"[{request_id}] Client disconnected, agent task continues "
            f"in background for session {session_id}"
        )

    except Exception as e:
        logger.error(f"[{request_id}] Streaming error: {e}", exc_info=True)
        friendly = "Something went wrong — please check the server and try again."
        yield f"data: {json.dumps({'error': friendly, 'session_id': session_id, 'done': True})}\n\n"


@router.post("/chat", response_model=ChatResponse)
async def chat_handler(
    req: ChatRequest,
    x_request_id: Optional[str] = Header(None)
) -> ChatResponse:
    """
    Process user message and return agent response
    
    - **session_id**: Unique session identifier (UUID)
    - **message**: User query to process
    
    Returns agent response with reasoning and data
    """
    request_id = x_request_id or str(uuid.uuid4())
    logger.info(f"[{request_id}] Chat request - Session: {req.session_id}")
    logger.debug(f"[{request_id}] Message: {req.message[:100]}...")
    
    try:
        selected_model = req.model or None  # None will use default from settings

        # Submit agent work as a background asyncio.Task.
        # The task survives client disconnects — if the user switches
        # sessions the response still completes and is saved to Memory.
        task = await submit_agent_task(
            req.session_id, req.message, model=selected_model
        )

        if req.stream:
            return StreamingResponse(
                stream_chat_response(task, req.session_id, request_id),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # disable nginx buffering
                },
            )

        # Non-streaming: await the background task directly
        logger.debug(f"[{request_id}] Awaiting agent task (model: {selected_model or 'default'})...")
        response_text = await task

        logger.info(f"[{request_id}] Chat completed successfully")
        return ChatResponse(
            content=response_text,
            session_id=req.session_id,
        )
        
    except NotFoundError as e:
        logger.warning(f"[{request_id}] Resource not found: {e}")
        raise HTTPException(
            status_code=404,
            detail=f"Resource not found: {str(e)}"
        )
    except APIError as e:
        logger.error(f"[{request_id}] External API error: {e}")
        raise HTTPException(
            status_code=503,
            detail="External API unavailable, please try again"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail="Something went wrong — please check the server and try again."
        )


@router.post("/session/new", response_model=SessionCreateResponse)
async def create_session(x_request_id: Optional[str] = Header(None)) -> SessionCreateResponse:
    """
    Create a new chat session
    
    Each session maintains its own conversation history
    and memory in the SQLite chat store
    """
    request_id = x_request_id or str(uuid.uuid4())
    new_session_id = str(uuid.uuid4())
    
    try:
        logger.info(f"[{request_id}] Creating new session: {new_session_id}")
        
        # Initialize session in database
        db = SessionLocal()

        session = ChatSession(
            id=new_session_id,
            title=f"Cinematic Discussion {new_session_id[:8]}"
        )
        db.add(session)
        db.commit()
        db.close()
        
        logger.info(f"[{request_id}] ✓ Session created: {new_session_id}")
        
        return SessionCreateResponse(session_id=new_session_id)
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to create session: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to create session"
        )


@router.get("/session/{session_id}")
async def get_session_info(
    session_id: str,
    x_request_id: Optional[str] = Header(None)
):
    """
    Retrieve session information
    
    Returns session metadata and conversation history count
    """
    request_id = x_request_id or str(uuid.uuid4())
    
    try:
        logger.debug(f"[{request_id}] Fetching session: {session_id}")
        
        db = SessionLocal()

        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        db.close()
        
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
        
        return {
            "session_id": session.id,
            "title": session.title,
            "created_at": session.created_at,
            "metadata": session.metadata_info
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error fetching session: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch session")


@router.delete("/session/{session_id}")
async def delete_session(
    session_id: str,
    x_request_id: Optional[str] = Header(None)
):
    """
    Delete a chat session and its history
    """
    request_id = x_request_id or str(uuid.uuid4())
    
    try:
        logger.info(f"[{request_id}] Deleting session: {session_id}")
        
        db = SessionLocal()

        # Cascade: delete messages first, then session
        db.query(ChatMessage).filter(ChatMessage.session_id == session_id).delete()
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            db.delete(session)
        db.commit()
        db.close()

        # Cancel any running background tasks for this session
        drop_session(session_id)
        
        logger.info(f"[{request_id}] Session deleted: {session_id}")
        
        return {"status": "deleted", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")


@router.get("/session/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    x_request_id: Optional[str] = Header(None),
):
    """
    Retrieve all messages for a session from the database.

    Returns the full conversation history ordered by creation time.
    Use this when the user switches back to a previous session so the
    frontend can display the entire chat.
    """
    request_id = x_request_id or str(uuid.uuid4())

    try:
        logger.debug(f"[{request_id}] Fetching messages for session: {session_id}")

        db = SessionLocal()
        rows = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .order_by(ChatMessage.created_at.asc())
            .all()
        )
        messages = [
            {
                "id": r.id,
                "role": r.role,
                "content": r.content,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]
        db.close()

        return {"session_id": session_id, "messages": messages}

    except Exception as e:
        logger.error(f"[{request_id}] Error fetching messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch messages")


@router.get("/sessions")
async def list_sessions(x_request_id: Optional[str] = Header(None)):
    """
    List all chat sessions with message counts.

    Used by the frontend to rebuild the conversation list from the DB
    on page load / refresh.
    """
    request_id = x_request_id or str(uuid.uuid4())
    try:
        db = SessionLocal()
        sessions = db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()
        result = []
        for s in sessions:
            msg_count = (
                db.query(ChatMessage)
                .filter(ChatMessage.session_id == s.id)
                .count()
            )
            result.append({
                "session_id": s.id,
                "title": s.title,
                "created_at": s.created_at.isoformat() if s.created_at else None,
                "message_count": msg_count,
            })
        db.close()
        return {"sessions": result}
    except Exception as e:
        logger.error(f"[{request_id}] Error listing sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to list sessions")


@router.delete("/session/{session_id}/messages")
async def clear_session_messages(
    session_id: str,
    x_request_id: Optional[str] = Header(None),
):
    """
    Delete all messages for a session (clear chat) without deleting
    the session itself.
    """
    request_id = x_request_id or str(uuid.uuid4())
    try:
        db = SessionLocal()
        deleted = (
            db.query(ChatMessage)
            .filter(ChatMessage.session_id == session_id)
            .delete()
        )
        db.commit()
        db.close()
        logger.info(f"[{request_id}] Cleared {deleted} messages for session {session_id}")
        return {"status": "cleared", "session_id": session_id, "deleted": deleted}
    except Exception as e:
        logger.error(f"[{request_id}] Error clearing messages: {e}")
        raise HTTPException(status_code=500, detail="Failed to clear messages")


@router.patch("/session/{session_id}")
async def update_session(
    session_id: str,
    body: dict,
    x_request_id: Optional[str] = Header(None),
):
    """
    Update session metadata (e.g. title).
    """
    request_id = x_request_id or str(uuid.uuid4())
    try:
        db = SessionLocal()
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if not session:
            db.close()
            raise HTTPException(status_code=404, detail="Session not found")
        if "title" in body:
            session.title = body["title"]
        db.commit()
        db.close()
        return {"status": "updated", "session_id": session_id}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Error updating session: {e}")
        raise HTTPException(status_code=500, detail="Failed to update session")


@router.get("/session/{session_id}/status")
async def get_session_status(
    session_id: str,
    x_request_id: Optional[str] = Header(None),
):
    """
    Check whether the agent has a running background task for this session.

    The frontend polls this to know when to refresh messages after the
    user navigates away from a chat that was still generating.
    """
    running = get_active_tasks(session_id)
    return {
        "session_id": session_id,
        "pending": len(running) > 0,
        "task_count": len(running),
    }


@router.post("/session/{session_id}/cancel")
async def cancel_session(
    session_id: str,
    x_request_id: Optional[str] = Header(None),
):
    """
    Cancel all running background tasks for this session.

    Called by the frontend when a timeout is reached — stops the agent
    so it doesn't keep consuming resources for a request the user has
    already abandoned.
    """
    cancelled = cancel_session_tasks(session_id)
    return {
        "session_id": session_id,
        "cancelled": cancelled,
    }