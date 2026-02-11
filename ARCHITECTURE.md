# Movie Analyst Agent — Architecture & Technical Documentation

> Single-document reference for the **backend** (FastAPI + LlamaIndex ReAct Agent) and **frontend** (Streamlit).

---

## Table of Contents

1. [High-Level Overview](#high-level-overview)
2. [Backend](#backend)
   - [Entry Point](#entry-point)
   - [API Routes](#api-routes)
   - [Agent Orchestration](#agent-orchestration)
   - [Streaming — Mimic Mode](#streaming--mimic-mode)
   - [Database Layer](#database-layer)
   - [Memory Hydration](#memory-hydration)
   - [Background Tasks & Disconnect Safety](#background-tasks--disconnect-safety)
   - [Tools (TMDB / TVMaze)](#tools-tmdb--tvmaze)
   - [LLM Adapter (Concentrate AI)](#llm-adapter-concentrate-ai)
   - [Configuration](#configuration)
3. [Frontend](#frontend)
   - [App Lifecycle](#app-lifecycle)
   - [Session & Message Flow](#session--message-flow)
   - [Auto-Refresh Poller](#auto-refresh-poller)
   - [Sidebar — Conversations, Rename, Delete](#sidebar--conversations-rename-delete)
   - [Streaming & Non-Streaming UX](#streaming--non-streaming-ux)
   - [Local vs DB State](#local-vs-db-state)
4. [Data Flow — End to End](#data-flow--end-to-end)
5. [Streaming — Why We Mimic It](#streaming--why-we-mimic-it)
6. [Docker Compose](#docker-compose)

---

## High-Level Overview

```
User ──▶ Streamlit (port 8501) ──HTTP/SSE──▶ FastAPI (port 8000)
                                                │
                                    ┌───────────┼───────────┐
                                    ▼           ▼           ▼
                              LlamaIndex    SQLite DB   Concentrate AI
                              ReAct Agent   (WAL mode)  (multi-LLM)
                              12 TMDB tools              ▲
                                    │                    │
                                    └── TMDB / TVMaze ───┘
```

The **backend** is a stateless FastAPI service. Every request hydrates a fresh
`Memory` from the SQLite database, creates a disposable `ReActAgent`, runs the
query, and persists the result back. Background `asyncio.Task`s ensure the
agent completes even if the frontend disconnects.

The **frontend** is a Streamlit app that talks to the backend over HTTP.
It stores **only local settings** (selected model, active session) on disk;
**all messages live in the backend database**. A `@st.fragment(run_every=3)`
poller checks for new messages and renders them **inline** (no full-page
rerun). A 90-second timeout cancels the backend task if it takes too long.

---

## Backend

### Entry Point

**`backend/main.py`** — Creates the Uvicorn ASGI server, calls `init_db()`,
and mounts the FastAPI application from `src/api/main_app.py`.

### API Routes

**`backend/src/api/routes.py`** — All endpoints live under `/api/v1/`.

| Method   | Path                             | Purpose |
|----------|----------------------------------|---------|
| `POST`   | `/chat`                          | Send a message to the agent. Accepts `stream: true` for SSE. |
| `POST`   | `/session/new`                   | Create a new chat session (UUID). |
| `GET`    | `/session/{id}`                  | Session metadata (title, created_at). |
| `DELETE` | `/session/{id}`                  | Delete session + cascade-delete all messages. |
| `GET`    | `/session/{id}/messages`         | Fetch all messages for a session (ordered by time). |
| `DELETE` | `/session/{id}/messages`         | Clear messages without deleting the session. |
| `GET`    | `/sessions`                      | List all sessions with message counts. |
| `PATCH`  | `/session/{id}`                  | Update session metadata (e.g. rename title). |
| `GET`    | `/session/{id}/status`           | Check if the agent has a running background task (`pending: true/false`). |
| `POST`   | `/session/{id}/cancel`           | Cancel all running background tasks for a session (used on timeout). |
| `GET`    | `/health`                        | Health check — `200 OK`. |

#### Chat endpoint details

```
POST /api/v1/chat
{
  "session_id": "uuid",
  "message": "What are the trending movies?",
  "model": "auto",       // optional — defaults to settings
  "stream": true          // optional — false for plain JSON response
}
```

When `stream: true`, the response is an SSE stream (`text/event-stream`).
Each event is `data: {"content": "...", "session_id": "..."}`. The final
event includes `"done": true`.

### Agent Orchestration

**`backend/src/services/agent.py`** — The core module. Key functions:

| Function | Role |
|----------|------|
| `submit_agent_task(session_id, message, model)` | Entry point. Hydrates memory from DB → **saves user message to DB immediately** → creates agent → launches `asyncio.Task`. Returns the task so callers can await or stream from it. |
| `_run_agent_task(session_id, agent, message, memory)` | Runs `agent.run()` inside the background task. **Saves only the assistant reply** to DB when done. |
| `_build_memory(session_id)` | Creates a fresh `Memory` object and populates it from all DB messages for the session. |
| `create_agent(model)` | Factory — creates a lightweight `ReActAgent` with the Concentrate AI LLM and 12 TMDB tools. |
| `get_active_tasks(session_id)` | Returns all in-flight `asyncio.Task`s for a session. Used by the `/status` endpoint. |
| `cancel_session_tasks(session_id)` | Cancels running tasks without removing session data. Used by the frontend timeout and the `/cancel` endpoint. Returns the number of tasks cancelled. |
| `drop_session(session_id)` | Cancels all running tasks for a session (called on delete). |

**Message saving is split into two phases:**

1. **Immediately** — The user message is saved to the DB inside
   `submit_agent_task()`, before the agent starts. This guarantees the prompt
   is persisted even if the agent crashes or the connection drops.

2. **On completion** — The assistant reply is saved inside `_run_agent_task()`
   after the agent finishes. Only one row is written at this stage.

### Streaming — Mimic Mode

**Concentrate AI's streaming API does not return chunks correctly** — responses
come back empty or fail to emit tokens. Because of this, the backend **mimics
streaming**:

1. The agent runs to completion inside the background task.
2. `stream_chat_response()` awaits the full result via `asyncio.shield(task)`.
3. The complete text is split into **3-word batches**.
4. Each batch is emitted as an SSE `data:` event with a **30 ms
   `asyncio.sleep`** between chunks, producing a natural typing effect.

```python
# routes.py — stream_chat_response()
words = final_text.split(' ')
chunk_size = 3
for i in range(0, len(words), chunk_size):
    batch = words[i:i + chunk_size]
    chunk = ' '.join(batch) + (' ' if i + chunk_size < len(words) else '')
    yield f"data: {json.dumps({'content': chunk, 'session_id': session_id})}\n\n"
    await asyncio.sleep(0.03)
```

The code is fully wired so that switching to real token-by-token streaming
requires only replacing this loop with the upstream stream. No frontend changes
needed.

### Database Layer

**`backend/src/core/db.py`** — SQLAlchemy + SQLite.

**Tables:**

| Table | Columns | Notes |
|-------|---------|-------|
| `chat_sessions` | `id` (PK, UUID), `title`, `created_at`, `updated_at`, `metadata_info` (JSON) | One row per conversation. |
| `chat_messages` | `id` (PK, UUID), `session_id` (indexed), `role`, `content`, `created_at` (indexed) | User and assistant messages. Ordered by `created_at`. |

**SQLite optimizations:**

- **WAL (Write-Ahead Logging)** mode — concurrent reads during writes.
- **PRAGMA synchronous=NORMAL** — balanced durability vs performance.
- `check_same_thread=False` — required for FastAPI's async model.

The DB file location is controlled by the `DATABASE_PATH` env var (default
`./chat_history.db`). In Docker, the `./data` volume is mounted so the DB
persists across container restarts.

### Memory Hydration

Every request builds a **fresh** `Memory` object from the database — there is
**zero in-memory caching**:

```python
async def _build_memory(session_id):
    _ensure_session_exists(session_id)
    history = _load_history(session_id)  # SELECT * FROM chat_messages ORDER BY created_at

    memory = Memory.from_defaults(
        session_id=session_id,
        token_limit=settings.AGENT_MEMORY_TOKEN_LIMIT,
    )
    for msg in history:
        memory.put(ChatMessage(role=msg["role"], content=msg["content"]))
    return memory
```

This means:
- No stale cache issues.
- Multiple backend instances can serve the same session (horizontal scaling).
- The trade-off is a DB read per request — negligible for SQLite.

### Background Tasks & Disconnect Safety

When a user sends a message:

1. `submit_agent_task()` saves the user message and creates an `asyncio.Task`.
2. The route wraps the task in `asyncio.shield()` before streaming.
3. If the user disconnects (switches tabs, closes browser), the SSE generator
   is cancelled — but `shield()` ensures the **task keeps running**.
4. The task finishes, writes the assistant reply to DB, and cleans itself up
   via a `done_callback`.

When the user comes back (or the frontend polls), the completed reply is
visible from the DB.

### Tools (TMDB / TVMaze)

**`backend/src/services/tool.py`** — 12 LlamaIndex `FunctionTool` instances:

| Tool | API | Description |
|------|-----|-------------|
| `search_tool` | TMDB | Universal search (movies, TV, people) |
| `details_tool` | TMDB | Full details — credits, recommendations |
| `trending_tool` | TMDB | Trending content (day/week) |
| `popular_tool` | TMDB | Popular movies or TV shows |
| `top_rated_tool` | TMDB | Highest-rated content |
| `upcoming_tool` | TMDB | Upcoming releases |
| `airing_today_tool` | TMDB/TVMaze | TV airing today |
| `on_the_air_tool` | TMDB | Currently airing series |
| `similar_tool` | TMDB | Similar movies/shows |
| `recommendations_tool` | TMDB | TMDB-powered recommendations |
| `discover_tool` | TMDB | Advanced filtering (genre, year, rating) |
| `find_id_tool` | TMDB | Look up by IMDb/TVDB external ID |

All tools use **HTTPX** (async) with **Tenacity** exponential backoff for
retries.

### LLM Adapter (Concentrate AI)

**`backend/src/services/concentrate_llm.py`** — A custom LlamaIndex `LLM`
subclass that adapts Concentrate AI's API (similar to OpenAI but not fully
compatible) for use with the ReAct agent.

Key behaviors:
- Sends requests to `https://api.concentrate.ai/v1`.
- Supports `tool_choice: "none"` to prevent unwanted auto-tool-calls.
- Configurable `max_output_tokens`.
- Falls back to non-streaming chat completion (because Concentrate AI streaming
  is broken — see [Streaming](#streaming--why-we-mimic-it)).

### Configuration

**`backend/src/core/config.py`** — Pydantic `BaseSettings` with `.env` support.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONCENTRATE_API_KEY` | *required* | Concentrate AI gateway key |
| `TMDB_API_KEY` | *required* | TMDB API key |
| `DEFAULT_MODEL` | `gpt-4o-mini` | LLM model (`auto`, `gpt-4`, `claude-3-5-sonnet-20241022`, etc.) |
| `DATABASE_PATH` | `./chat_history.db` | SQLite file location |
| `DATABASE_POOL_SIZE` | `5` | Connection pool size |
| `AGENT_MEMORY_TOKEN_LIMIT` | `4000` | Max tokens in agent memory |
| `MAX_OUTPUT_TOKENS` | `1000` | Max output tokens per response |
| `HTTP_TIMEOUT` | `60` | Request timeout (seconds) |
| `HTTP_RETRIES` | `3` | Retry attempts |
| `LOG_LEVEL` | `INFO` | Logging verbosity |
| `ALLOWED_ORIGINS` | `localhost:8501` | CORS origins |

---

## Frontend

### App Lifecycle

**`frontend/app.py`** — A single-file Streamlit application.

On first load:
1. `_load_conversations()` reads local settings from `conversations.json`
   (model, streaming toggle, last active session).
2. Calls `GET /sessions` to fetch all sessions from the backend DB.
3. Calls `GET /session/{id}/messages` for the active session to load messages.
4. Renders the sidebar (conversation list) and main chat area.

### Session & Message Flow

```
User types prompt
        │
        ▼
Append user msg to local state
        │
        ╠═══▶  PATCH /session/{id}  (update title if "New Chat")
        │
        ▼
send_message() ──▶ POST /api/v1/chat ──▶ Backend
        │                                     │
        │                              submit_agent_task()
        │                              ├── Save user msg to DB (immediate)
        │                              └── Launch asyncio.Task
        │                                     │
        │                              _run_agent_task()
        │                              ├── ReAct agent runs
        │                              └── Save assistant msg to DB
        │                                     │
        ◄─── SSE chunks (mimicked streaming) ─┘
        │
Append assistant msg to local state
```

If the user **navigates away** before the response finishes:
- The backend task continues (shield).
- The `@st.fragment` poller detects the new message and refreshes automatically.

### Auto-Refresh Poller (Inline, No Full Rerun)

```python
@st.fragment(run_every=3)
def _poll_for_updates():
    # Last message is already from assistant → no-op
    if last["role"] == "assistant":
        return

    # Poll DB for new messages
    db_msgs = fetch_messages(sid)
    if len(db_msgs) > len(active["messages"]):
        # Render new reply INLINE inside the fragment
        active["messages"] = db_msgs
        with st.chat_message("assistant", avatar="🎬"):
            st.markdown(db_msgs[-1]["content"])
        return

    # Timeout reached → cancel backend task
    if elapsed >= _AGENT_TIMEOUT_SECONDS:
        cancel_agent_task(sid)
        with st.chat_message("assistant", avatar="🎬"):
            st.warning("⏱️ The agent took too long and was stopped.")
        return

    # Still waiting → show elapsed time
    with st.chat_message("assistant", avatar="🎬"):
        st.markdown(f"⏳ *Thinking… ({secs}s)*")
```

**How it works:**

- `@st.fragment(run_every=3)` re-executes only this function every 3 seconds
  — the rest of the page is **not** rerun.
- The fragment renders output **inside its own boundary**: the thinking
  indicator, the completed response, or the timeout message all appear in
  the same spot, replacing each other seamlessly.
- **No `st.rerun()`** — the fragment swap is enough. The page doesn't flash
  or scroll.
- A **timeout** (default 90 s) is tracked via `st.session_state`. When
  reached, `POST /session/{id}/cancel` tells the backend to kill the
  `asyncio.Task`, and a timeout warning replaces the thinking indicator.
- When the last message is already from the assistant the function exits
  immediately — zero network calls, zero renders.

### Sidebar — Conversations, Rename, Delete

- Each conversation shows a **select button** + a popover (`⋮`) with
  **✏️ Rename** and **🗑️ Delete** actions.
- Rename calls `PATCH /session/{id}` to sync the title to the backend.
- Delete calls `DELETE /session/{id}` which cascade-deletes messages.
- Conversation count is derived from local messages or `message_count` from
  the sessions list.

### Streaming & Non-Streaming UX

| Mode | Behavior |
|------|----------|
| **Streaming** (`use_streaming=True`) | `send_message()` opens an SSE connection. Each `data:` chunk is appended to an `st.empty()` placeholder with a `▌` cursor. Final text replaces the cursor. |
| **Non-streaming** (`use_streaming=False`) | `send_message()` posts to `/chat` with `stream: false`. Shows `st.spinner("Analyzing…")` while waiting. Renders the full response on completion. |

### Local vs DB State

| Data | Where it lives | Why |
|------|----------------|-----|
| Messages | **Backend SQLite DB** (single source of truth) | Survives page refreshes, browser restarts, container restarts. |
| Selected model, streaming toggle | `frontend/data/conversations.json` | Lightweight local settings — not worth a DB round-trip. |
| Active conversation ID | `conversations.json` | Remembers which chat the user was in. |
| Session list + titles | Backend DB | Frontend fetches on load. |

On conversation switch, the frontend uses a **merge strategy**: it keeps
whichever is longer — the DB messages or local messages — to handle the race
condition where a background task hasn't flushed to DB yet.

---

## Data Flow — End to End

```
1. User opens app → frontend calls GET /sessions → builds sidebar
2. User clicks a conversation → frontend calls GET /session/{id}/messages
3. User types a prompt → frontend calls POST /api/v1/chat (stream=true)
4. Backend:
   a. Hydrates Memory from DB (_build_memory)
   b. Saves user message to DB immediately
   c. Creates ReActAgent
   d. Launches asyncio.Task (_run_agent_task)
   e. SSE generator awaits task via asyncio.shield()
   f. Agent queries TMDB tools, reasons via Concentrate AI LLM
   g. Final response split into 3-word chunks, emitted with 30ms delay
   h. Assistant message saved to DB
5. Frontend appends assistant message to local state
6. If user navigated away during step 4:
   - Backend task still completes (shield)
   - Frontend @st.fragment polls DB every 3s (inline, no page reload)
   - Detects new message → renders reply inside the fragment boundary
   - If 90s timeout reached → POST /session/{id}/cancel → kills task
   - Timeout warning replaces thinking indicator
```

---

## Streaming — Why We Mimic It

**Concentrate AI's streaming endpoint is broken** — it returns empty chunks
or fails to emit tokens entirely (see issue #5 in the known shortcomings).

Because of this, the backend uses a **mimic streaming** pattern:

1. The agent runs **synchronously to completion** inside a background task.
2. The full response text is captured.
3. `stream_chat_response()` splits the text into **3-word batches**.
4. Each batch is emitted as an SSE `data:` event with a **30 ms sleep**
   between events.
5. The frontend reads these events and renders them progressively.

**From the user's perspective**, it looks exactly like real streaming — text
appears word-by-word in real time. The only difference is a slightly longer
initial delay (the agent must finish before any text appears), rather than
seeing the first tokens while the model is still generating.

**The codebase is fully wired for real streaming.** When Concentrate AI fixes
their streaming API, the only change needed is replacing the word-batching loop
in `stream_chat_response()` with an async iterator over the upstream SSE
stream. No frontend changes required.

---

## Docker Compose

```yaml
services:
  backend:
    build: ./backend
    ports: ["8000:8000"]
    env_file: ./backend/.env
    volumes: ["./data:/app/data"]    # SQLite DB persists here
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s

  frontend:
    build: ./frontend
    ports: ["8501:8501"]
    environment:
      BACKEND_URL: http://backend:8000
    depends_on:
      backend:
        condition: service_healthy
```

- The frontend waits for the backend health check before starting.
- The `./data` volume ensures the SQLite DB survives container restarts.
- Log rotation: JSON driver, 10 MB × 3 files.
- Both services restart automatically unless explicitly stopped.
