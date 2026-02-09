import asyncio
from fastapi import APIRouter, HTTPException, Header
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
import uuid
import json
from typing import Optional, AsyncGenerator

from src.services.agent import get_agent_for_session
from src.schemas.chat import (
    ChatRequest,
    ChatResponse,
    SessionCreateResponse,
    ErrorResponse
)
from src.core.logger import get_logger
from src.core.db import SessionLocal
from src.services.tool import NotFoundError, APIError
from llama_index.core.workflow import Context


logger = get_logger("routes")
router = APIRouter()


async def stream_chat_response(
    agent, 
    message: str, 
    session_id: str, 
    request_id: str,
    memory
) -> AsyncGenerator[str, None]:
    """Stream agent response as SSE events.
    
    ReActAgent emits workflow events (AgentStream for text deltas,
    ToolCallResult for tool outputs, etc.) via stream_events().
    We forward text deltas to the client. If no deltas are emitted
    """
    try:
        logger.info(f"[{request_id}] Starting streaming response")
        sent_any = False
        
        ctx = Context(agent)
        handler = agent.run(message, memory=memory, ctx=ctx, max_iterations=10)
        
        # async for event in handler.stream_events():
        #     evt_type = type(event).__name__
            
        #     # AgentStream events carry text deltas
        #     if hasattr(event, 'delta') and event.delta:
        #         delta = str(event.delta)
        #         if delta.strip():
        #             sent_any = True
        #             yield f"data: {json.dumps({'content': delta, 'session_id': session_id})}\n\n"
            
            # # ToolCallResult — optionally notify the client that a tool ran
            # elif hasattr(event, 'tool_name') and hasattr(event, 'tool_output'):
            #     tool_msg = f"\n🔧 *Used {event.tool_name}*\n"
            #     yield f"data: {json.dumps({'content': tool_msg, 'session_id': session_id})}\n\n"
            #     sent_any = True
        
        # Get the final aggregated response
        final_response = await handler
        final_text = str(final_response).strip()
        logger.debug(f"[{request_id}] Final response ({len(final_text)} chars): {final_text[:120]}...")

        #    if not sent_any and final_text:
        #     # No deltas emitted — send the whole answer
        #     yield f"data: {json.dumps({'content': final_text, 'session_id': session_id, 'done': True})}\n\n"
        # else:
        #     yield f"data: {json.dumps({'session_id': session_id, 'done': True})}\n\n"
        

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
        logger.info(f"[{request_id}] ✓ Streaming completed")
        
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
        agent, memory = await get_agent_for_session(req.session_id, model=selected_model)
        
        if req.stream:
            return StreamingResponse(
                stream_chat_response(agent, req.message, req.session_id, request_id, memory),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",  # disable nginx buffering
                },
            )
        
        # Run agent with message (non-streaming)
        logger.debug(f"[{request_id}] Invoking agent (model: {selected_model or 'default'})...")
        handler = agent.run(req.message, memory=memory, max_iterations=10)
        response = await handler
        
        response_text = str(response).strip()
        
        chat_response = ChatResponse(
            content=response_text,
            session_id=req.session_id
        )
        
        logger.info(f"[{request_id}] ✓ Chat completed successfully")
        return chat_response
        
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
        from ..core.db import ChatSession
        
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
        from ..core.db import ChatSession
        
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
        from ..core.db import ChatSession
        
        session = db.query(ChatSession).filter(ChatSession.id == session_id).first()
        if session:
            db.delete(session)
            db.commit()
        
        db.close()
        
        logger.info(f"[{request_id}] ✓ Session deleted: {session_id}")
        
        return {"status": "deleted", "session_id": session_id}
        
    except Exception as e:
        logger.error(f"[{request_id}] Failed to delete session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete session")