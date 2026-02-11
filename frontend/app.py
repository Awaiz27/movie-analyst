"""
Movie Analyst Agent — Streamlit Frontend
"""

import streamlit as st
import requests
import os
import uuid
import logging
import json
import time
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional
 
logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(name)s  %(levelname)s  %(message)s")
logger = logging.getLogger("frontend")

API_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
API_BASE = f"{API_URL}/api/v1"
BACKEND_HEALTH_CHECK = f"{API_URL}/health"
CONCENTRATE_PROVIDERS_URL = "https://api.concentrate.ai/v1/models/"
_DATA_DIR = Path(__file__).parent / "data"
_DATA_DIR.mkdir(exist_ok=True)
_CONV_FILE = _DATA_DIR / "conversations.json"

FALLBACK_MODELS = {
    "auto": "Auto (Smart Routing)",
    "gpt-4": "GPT-4 (OpenAI)",
    "gpt-3.5-turbo": "GPT-3.5 Turbo",
    "claude-3-5-sonnet-20241022": "Claude 3.5 Sonnet",
    "claude-3-opus-20240229": "Claude 3 Opus",
    "gemini-pro": "Gemini Pro (Google)",
    "gemini-1.5-pro": "Gemini 1.5 Pro",
}

QUICK_PROMPTS = [
    ("🔥", "What's trending?", "What are the trending movies right now?"),
    ("⭐", "Top rated", "What are the top rated movies of all time?"),
    ("🎭", "Now playing", "What movies are now playing in theaters?"),
    ("📺", "Popular TV", "What are the most popular TV shows right now?"),
]

LANDING_PROMPTS = [
    ("🔥", "Trending Movies", "What are the trending movies right now?"),
    ("⭐", "Top Rated", "What are the top rated movies of all time?"),
    ("🎭", "Now Playing", "What movies are now playing in theaters?"),
    ("📺", "TV Schedules", "What popular TV shows are airing this week?"),
    ("🎯", "Recommendations", "Can you recommend some great movies to watch?"),
    ("🎬", "Cast & Crew", "Tell me about notable directors and their best films"),
]

st.set_page_config(
    page_title="Movie Analyst Agent",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded",
)


_CSS_PATH = Path(__file__).parent / "static" / "style.css"
st.markdown(f"<style>{_CSS_PATH.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

def _save_conversations() -> None:
    """Persist local settings to disk. Messages live in the backend DB."""
    try:
        payload = {
            "active_conv": st.session_state.active_conv,
            "selected_model": st.session_state.selected_model,
            "use_streaming": st.session_state.use_streaming,
        }
        _CONV_FILE.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save settings: %s", exc)


def _load_conversations() -> None:
    """Load sessions from backend DB + local settings from disk."""
    # ── Local settings (model, streaming, last active session) ──
    local = {}
    if _CONV_FILE.exists():
        try:
            local = json.loads(_CONV_FILE.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Failed to load settings file: %s", exc)

    st.session_state.selected_model = local.get("selected_model", "auto")
    st.session_state.use_streaming = local.get("use_streaming", True)

    # ── Build conversation list from backend DB ──
    sessions = fetch_sessions()
    conversations = {}
    for s in sessions:
        sid = s["session_id"]
        conversations[sid] = {
            "session_id": sid,
            "messages": [],
            "title": s.get("title", "New Chat"),
            "created_at": s.get("created_at", ""),
            "model": st.session_state.selected_model,
            "message_count": s.get("message_count", 0),
        }

    st.session_state.conversations = conversations

    # Restore active conversation from last session
    prev_active = local.get("active_conv")
    if prev_active and prev_active in conversations:
        st.session_state.active_conv = prev_active
    elif conversations:
        st.session_state.active_conv = next(iter(conversations))
    else:
        st.session_state.active_conv = None

    # Load messages for the active conversation from the DB
    if st.session_state.active_conv:
        sid = st.session_state.active_conv
        db_msgs = fetch_messages(sid)
        local_msgs = conversations[sid].get("messages", [])
        conversations[sid]["messages"] = (
            db_msgs if len(db_msgs) >= len(local_msgs) else local_msgs
        )
        msgs = conversations[sid]["messages"]
        # Derive title from first user message if still generic
        if msgs and conversations[sid]["title"] in ("New Chat", f"New Chat {sid[:8]}"):
            conversations[sid]["title"] = _auto_title(msgs)


@st.cache_data(ttl=30, show_spinner=False)
def check_backend_health() -> bool:
    """Ping the backend health endpoint (result cached 30 s)."""
    try:
        return requests.get(BACKEND_HEALTH_CHECK, timeout=3).status_code == 200
    except Exception as exc:
        logger.warning("Backend health check failed: %s", exc)
        return False


@st.cache_data(ttl=3600, show_spinner=False)
def load_concentrate_providers() -> dict[str, str]:
    """Fetch available models from Concentrate AI; fall back to built-in list."""
    try:
        resp = requests.get(CONCENTRATE_PROVIDERS_URL, timeout=10)
        resp.raise_for_status()
        providers = resp.json()
        if isinstance(providers, list) and providers:
            pmap = {
                p.get("slug", ""): p.get("name", p.get("slug", ""))
                for p in providers if p.get("slug")
            }
            if pmap:
                return {"auto": "Auto (Smart Routing)", **pmap}
    except Exception as exc:
        logger.warning("Provider fetch failed, using fallback: %s", exc)
    return FALLBACK_MODELS


def create_new_session() -> Optional[str]:
    """Create a new chat session on the backend."""
    try:
        resp = requests.post(
            f"{API_BASE}/session/new",
            timeout=10,
            headers={"x-request-id": str(uuid.uuid4())},
        )
        resp.raise_for_status()
        sid = resp.json().get("session_id")
        logger.info("Session created: %s", sid)
        return sid
    except requests.exceptions.ConnectionError:
        st.error("Cannot reach the backend — is the service running?")
    except requests.exceptions.RequestException as exc:
        st.error(f"Session creation failed: {exc}")
        logger.error("Session creation error: %s", exc)
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        logger.error("Unexpected error in create_new_session: %s", exc)
    return None


def fetch_sessions() -> list[dict]:
    """Fetch all sessions from the backend DB."""
    try:
        resp = requests.get(f"{API_BASE}/sessions", timeout=10)
        if resp.status_code == 200:
            return resp.json().get("sessions", [])
    except Exception as exc:
        logger.warning("Failed to fetch sessions: %s", exc)
    return []


def fetch_messages(session_id: str) -> list[dict]:
    """Fetch all messages for a session from the backend DB."""
    try:
        resp = requests.get(
            f"{API_BASE}/session/{session_id}/messages",
            timeout=10,
        )
        if resp.status_code == 200:
            msgs = resp.json().get("messages", [])
            return [{"role": m["role"], "content": m["content"]} for m in msgs]
    except Exception as exc:
        logger.warning("Failed to fetch messages for %s: %s", session_id, exc)
    return []


def check_session_pending(session_id: str) -> bool:
    """Return True if the backend has a running agent task for this session."""
    try:
        resp = requests.get(
            f"{API_BASE}/session/{session_id}/status",
            timeout=5,
        )
        if resp.status_code == 200:
            return resp.json().get("pending", False)
    except Exception as exc:
        logger.warning("Failed to check status for %s: %s", session_id, exc)
    return False


def cancel_agent_task(session_id: str) -> None:
    """Ask the backend to cancel running tasks for a session (timeout)."""
    try:
        requests.post(f"{API_BASE}/session/{session_id}/cancel", timeout=5)
    except Exception as exc:
        logger.warning("Failed to cancel task for %s: %s", session_id, exc)


# Maximum seconds to wait for the agent before showing a timeout message
_AGENT_TIMEOUT_SECONDS = 90


def _fire_agent_request(
    session_id: str, message: str, model: str
) -> None:
    """Send POST /chat in a background thread (fire-and-forget).

    The backend saves the user message to DB immediately and launches the
    agent as a background task. The frontend poller will pick up the
    assistant reply from the DB when it's ready — no need to block here.
    """
    try:
        requests.post(
            f"{API_BASE}/chat",
            json={"session_id": session_id, "message": message, "model": model, "stream": False},
            timeout=180,
            headers={"x-request-id": str(uuid.uuid4())},
        )
    except Exception as exc:
        logger.warning("Background agent request failed for %s: %s", session_id, exc)


def send_message(
    session_id: str,
    message: str,
    model: str = "auto",
    use_streaming: bool = False,
) -> Optional[str]:
    """Send a user message and return the assistant reply.

    When *use_streaming* is True the response is written token-by-token
    into a ``st.empty()`` placeholder for real-time feedback.
    """
    try:
        logger.info("msg → %s  model=%s  stream=%s", session_id, model, use_streaming)

        if use_streaming:
            resp = requests.post(
                f"{API_BASE}/chat",
                json={"session_id": session_id, "message": message, "model": model, "stream": True},
                timeout=180,
                headers={"x-request-id": str(uuid.uuid4())},
                stream=True,
            )
            if resp.status_code != 200:
                st.error(f"Backend returned {resp.status_code}")
                return None

            full = ""
            ph = st.empty()
            for raw_line in resp.iter_lines():
                if not raw_line:
                    continue
                text = raw_line.decode("utf-8")
                if not text.startswith("data: "):
                    continue
                try:
                    data = json.loads(text[6:])
                except json.JSONDecodeError:
                    continue
                if "error" in data:
                    st.error(f"Agent error: {data['error']}")
                    return None
                if "content" in data:
                    full += data["content"]
                    ph.markdown(full + " ▌")
                if data.get("done"):
                    break
            ph.markdown(full)
            return full

        else:
            resp = requests.post(
                f"{API_BASE}/chat",
                json={"session_id": session_id, "message": message, "model": model, "stream": False},
                timeout=180,
                headers={"x-request-id": str(uuid.uuid4())},
            )
            if resp.status_code == 200:
                return resp.json().get("content", "")
            if resp.status_code == 404:
                st.warning("Movie / show not found — try another query.")
            elif resp.status_code == 503:
                st.warning("Service temporarily unavailable. Try again shortly.")
            else:
                st.error(f"Error {resp.status_code}: {resp.text}")
            return None

    except requests.exceptions.Timeout:
        st.error("Request timed out. The agent may still be working — please retry.")
    except requests.exceptions.ConnectionError:
        st.error("Lost connection to the backend.")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
        logger.error("send_message error: %s", exc, exc_info=True)
    return None


def _auto_title(messages: list[dict]) -> str:
    """Derive a short conversation title from the first user message."""
    for m in messages:
        if m["role"] == "user":
            txt = m["content"].strip()
            return (txt[:40] + "…") if len(txt) > 40 else txt
    return "New Chat"


if "conversations" not in st.session_state:
    _load_conversations()
if "pending_prompt" not in st.session_state:
    st.session_state.pending_prompt = None


def _create_conv_with_prompt(prompt_text: str = None):
    """Create a new conversation and optionally set a pending prompt."""
    sid = create_new_session()
    if sid:
        st.session_state.conversations[sid] = {
            "session_id": sid,
            "messages": [],
            "title": "New Chat",
            "created_at": datetime.now().strftime("%b %d, %H:%M"),
            "model": st.session_state.selected_model,
            "message_count": 0,
        }
        st.session_state.active_conv = sid
        if prompt_text:
            st.session_state.pending_prompt = prompt_text
        _save_conversations()
        st.rerun()


# Convenience accessor
def _active() -> Optional[dict]:
    cid = st.session_state.active_conv
    return st.session_state.conversations.get(cid) if cid else None


# ---------------------------------------------------------------------------
# Auto-refresh poller — renders inline, replaces thinking with response
# ---------------------------------------------------------------------------
@st.fragment(run_every=3)
def _poll_for_updates():
    """
    Lightweight fragment that re-executes every 3 seconds.

    It renders the "thinking" indicator OR the completed assistant response
    inside its own boundary — no full-page st.rerun().

    Timeout: after _AGENT_TIMEOUT_SECONDS the backend task is cancelled
    and a timeout message is shown.
    """
    active = _active()
    if not active or not active.get("messages"):
        return

    last = active["messages"][-1]

    # Already have the assistant reply — render it and stop
    if last["role"] == "assistant":
        return

    # --- We are waiting for the assistant ---
    sid = active["session_id"]

    # Check how long we've been waiting
    wait_key = f"_wait_start_{sid}"
    if wait_key not in st.session_state:
        st.session_state[wait_key] = time.time()
    elapsed = time.time() - st.session_state[wait_key]

    # Poll DB for new messages
    db_msgs = fetch_messages(sid)

    if len(db_msgs) > len(active["messages"]):
        # The assistant reply arrived — update local state
        active["messages"] = db_msgs
        _save_conversations()
        st.session_state.pop(wait_key, None)
        # Single rerun so the main message loop picks up the new reply.
        st.rerun()
        return

    # Timeout reached — cancel the background task and show error
    if elapsed >= _AGENT_TIMEOUT_SECONDS:
        cancel_agent_task(sid)
        timeout_msg = (
            "⏱️ The agent took too long to respond and was stopped. "
            "Please try again."
        )
        active["messages"].append({"role": "assistant", "content": timeout_msg})
        _save_conversations()
        st.session_state.pop(wait_key, None)
        # Single rerun so the timeout message renders in the main loop
        st.rerun()
        return

    # Still waiting — show thinking indicator with elapsed time
    secs = int(elapsed)
    with st.chat_message("assistant", avatar="🎬"):
        st.markdown(f"⏳ *Thinking… ({secs}s)*")

backend_ok = check_backend_health()



with st.sidebar:
    # Brand
    st.markdown(
        "<div style='text-align:center;padding:1rem 0 .4rem'>"
        "<span style='font-size:2rem'>🎬</span><br>"
        "<span style='font-size:1.15rem;font-weight:700;letter-spacing:-.01em'>Movie Analyst</span><br>"
        "<span style='font-size:.65rem;color:rgba(255,255,255,.35);letter-spacing:.08em;"
        "text-transform:uppercase'>AI-Powered Movie Analyst</span>"
        "</div>",
        unsafe_allow_html=True,
    )

    dot_cls = "online" if backend_ok else "offline"
    dot_lbl = "Connected" if backend_ok else "Offline"
    st.markdown(
        f"<div class='status-row' style='justify-content:center'>"
        f"<span class='status-dot {dot_cls}'></span>{dot_lbl}</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # New conversation
    if st.button("＋  New Conversation", use_container_width=True, key="new_conv"):
        sid = create_new_session()
        if sid:
            st.session_state.conversations[sid] = {
                "session_id": sid,
                "messages": [],
                "title": "New Chat",
                "created_at": datetime.now().strftime("%b %d, %H:%M"),
                "model": st.session_state.selected_model,
                "message_count": 0,
            }
            st.session_state.active_conv = sid
            _save_conversations()
            st.rerun()

    # Conversation list
    convos = st.session_state.conversations
    if convos:
        st.markdown("<div class='sb-heading'>Conversations</div>", unsafe_allow_html=True)
        for cid in reversed(list(convos.keys())):
            c = convos[cid]
            is_active = cid == st.session_state.active_conv
            n_msgs = (
                len([m for m in c["messages"] if m["role"] == "user"])
                if c["messages"]
                else c.get("message_count", 0) // 2
            )
            title = c.get("title", "New Chat")

            # ── Row: [select button] [actions popover] ──
            col_sel, col_act = st.columns([9, 1])
            with col_sel:
                prefix = "\u25b8 " if is_active else "  "
                line = f"{prefix}{title}  ({n_msgs})"
                if st.button(
                    line,
                    key=f"conv_{cid}",
                    use_container_width=True,
                ):
                    st.session_state.active_conv = cid
                    # Fetch from DB, but keep local copy if it has more
                    # messages (e.g. background task hasn't flushed yet)
                    db_msgs = fetch_messages(c["session_id"])
                    local_msgs = convos[cid].get("messages", [])
                    convos[cid]["messages"] = (
                        db_msgs if len(db_msgs) >= len(local_msgs) else local_msgs
                    )
                    _save_conversations()
                    st.rerun()
            with col_act:
                with st.popover("⋮", use_container_width=True):
                    if st.button("✏️ Rename", key=f"ren_{cid}", use_container_width=True):
                        st.session_state["renaming_conv"] = cid
                        st.rerun()
                    if st.button("🗑️ Delete", key=f"del_{cid}", use_container_width=True):
                        try:
                            requests.delete(f"{API_BASE}/session/{cid}", timeout=10)
                        except Exception:
                            pass
                        del st.session_state.conversations[cid]
                        if st.session_state.active_conv == cid:
                            remaining = list(st.session_state.conversations.keys())
                            st.session_state.active_conv = remaining[-1] if remaining else None
                        _save_conversations()
                        st.rerun()

        renaming = st.session_state.get("renaming_conv")
        if renaming and renaming in convos:
            st.markdown("---")
            new_title = st.text_input(
                "Rename conversation",
                value=convos[renaming].get("title", ""),
                key="rename_input",
                max_chars=60,
            )
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("Save", key="rename_save", use_container_width=True):
                    new_title = new_title.strip()
                    if new_title:
                        convos[renaming]["title"] = new_title
                        try:
                            requests.patch(
                                f"{API_BASE}/session/{renaming}",
                                json={"title": new_title},
                                timeout=5,
                            )
                        except Exception:
                            pass
                    st.session_state.pop("renaming_conv", None)
                    _save_conversations()
                    st.rerun()
            with rc2:
                if st.button("Cancel", key="rename_cancel", use_container_width=True):
                    st.session_state.pop("renaming_conv", None)
                    st.rerun()
    else:
        st.markdown(
            "<div style='padding:1rem .4rem;text-align:center;color:rgba(255,255,255,.3);"
            "font-size:.82rem'>No conversations yet.<br>Start one above!</div>",
            unsafe_allow_html=True,
        )

    # Settings (when a conversation is active)
    active = _active()
    if active:
        st.divider()

        st.markdown("<div class='sb-heading'>Model</div>", unsafe_allow_html=True)
        model_opts = load_concentrate_providers()
        model_keys = list(model_opts.keys())
        cur = st.session_state.selected_model
        if cur not in model_opts:
            cur = "auto"
        st.session_state.selected_model = st.selectbox(
            "Model",
            options=model_keys,
            format_func=lambda k: model_opts.get(k, k),
            index=model_keys.index(cur),
            label_visibility="collapsed",
        )

        st.session_state.use_streaming = st.toggle(
            "Stream responses",
            value=st.session_state.use_streaming,
            help="Show tokens in real-time as the model generates them.",
        )

        st.divider()

        st.markdown(
            f"<div class='meta-chip'>"
            f"<div class='label'>Session</div>"
            f"<div class='value'>{active['session_id'][:16]}…</div>"
            f"</div>"
            f"<div class='meta-chip'>"
            f"<div class='label'>Created</div>"
            f"<div class='value'>{active.get('created_at', '—')}</div>"
            f"</div>"
            f"<div class='meta-chip'>"
            f"<div class='label'>Messages</div>"
            f"<div class='value'>{len(active['messages'])}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

        st.markdown("<div style='height:.5rem'></div>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            with st.container():
                st.markdown("<div class='secondary-btn'>", unsafe_allow_html=True)
                if st.button("Clear Chat", use_container_width=True, key="clear_chat"):
                    try:
                        requests.delete(
                            f"{API_BASE}/session/{active['session_id']}/messages",
                            timeout=10,
                        )
                    except Exception:
                        pass
                    active["messages"] = []
                    active["title"] = "New Chat"
                    _save_conversations()
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)
        with col_b:
            with st.container():
                st.markdown("<div class='secondary-btn'>", unsafe_allow_html=True)
                if st.button("Delete Chat", use_container_width=True, key="del_chat"):
                    cid = st.session_state.active_conv
                    try:
                        requests.delete(
                            f"{API_BASE}/session/{cid}",
                            timeout=10,
                        )
                    except Exception:
                        pass
                    del st.session_state.conversations[cid]
                    remaining = list(st.session_state.conversations.keys())
                    st.session_state.active_conv = remaining[-1] if remaining else None
                    _save_conversations()
                    st.rerun()
                st.markdown("</div>", unsafe_allow_html=True)



active = _active()

if active is not None:
    pending = st.session_state.get("pending_prompt")
    if pending:
        st.session_state.pending_prompt = None

    has_messages = len(active["messages"]) > 0

    if has_messages or pending:
        st.markdown("<div style='height:.6rem'></div>", unsafe_allow_html=True)

        for msg in active["messages"]:
            with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🎬"):
                st.markdown(msg["content"])

        # ── The fragment handles thinking indicator + auto-update inline ──
        _poll_for_updates()

        # Disable input while waiting for the agent to respond
        waiting = (
            bool(active["messages"])
            and active["messages"][-1]["role"] == "user"
        )
        user_input = st.chat_input(
            "Waiting for response…" if waiting else "Ask about any movie, show, or cinematic topic…",
            disabled=waiting,
        )
        prompt = pending or user_input

        if prompt:
            active["messages"].append({"role": "user", "content": prompt})
            if active["title"] == "New Chat":
                active["title"] = _auto_title(active["messages"])
                try:
                    requests.patch(
                        f"{API_BASE}/session/{active['session_id']}",
                        json={"title": active["title"]},
                        timeout=5,
                    )
                except Exception:
                    pass

            # Start the wait timer so the poller shows "Thinking…" immediately
            wait_key = f"_wait_start_{active['session_id']}"
            st.session_state[wait_key] = time.time()

            # Fire backend request in a background thread (non-blocking).
            # The backend saves the user msg to DB and starts the agent.
            # The poller will pick up the assistant reply when ready.
            threading.Thread(
                target=_fire_agent_request,
                args=(active["session_id"], prompt, st.session_state.selected_model),
                daemon=True,
            ).start()

            _save_conversations()
            st.rerun()  # Immediately re-render → poller shows "Thinking…"

    else:
        st.markdown("<div style='height:18vh'></div>", unsafe_allow_html=True)

        st.markdown(
            "<div style='text-align:center;margin-bottom:1.5rem'>"
            "<span style='font-size:2.5rem'>🎬</span><br>"
            "<span style='font-size:1.4rem;font-weight:600;color:var(--text-primary)'>What would you like to know?</span><br>"
            "<span style='font-size:.9rem;color:var(--text-secondary)'>Ask about any movie, TV show, actor, or cinematic topic</span>"
            "</div>",
            unsafe_allow_html=True,
        )

        # Clickable suggestion pills
        with st.container(key="pill_buttons_conv"):
            pill_cols = st.columns(len(QUICK_PROMPTS))
            for i, (icon, label, prompt_text) in enumerate(QUICK_PROMPTS):
                with pill_cols[i]:
                    if st.button(f"{icon} {label}", key=f"qp_{i}", use_container_width=True):
                        st.session_state.pending_prompt = prompt_text
                        st.rerun()

        if prompt := st.chat_input("Ask about any movie, show, or cinematic topic…"):
            st.session_state.pending_prompt = prompt
            st.rerun()

else:
    st.markdown(
        "<div class='hero-card'>"
        "<span class='hero-badge'>AI Agent</span>"
        "<h1>Movie Analyst Agent</h1>"
        "<div class='hero-sub'>Your intelligent research assistant for everything cinema.<br>"
        "Explore movies, TV shows, ratings, trends &amp; more.</div>"
        "</div>",
        unsafe_allow_html=True,
    )

    # Clickable pills — each creates a new conversation with the prompt
    with st.container(key="pill_buttons_landing"):
        lp_row1 = st.columns(3)
        for i, (icon, label, prompt_text) in enumerate(LANDING_PROMPTS[:3]):
            with lp_row1[i]:
                if st.button(f"{icon} {label}", key=f"lp_{i}", use_container_width=True):
                    _create_conv_with_prompt(prompt_text)
        lp_row2 = st.columns(3)
        for i, (icon, label, prompt_text) in enumerate(LANDING_PROMPTS[3:]):
            with lp_row2[i]:
                if st.button(f"{icon} {label}", key=f"lp_{i+3}", use_container_width=True):
                    _create_conv_with_prompt(prompt_text)

    # Feature cards
    st.markdown("<div class='feature-grid'>", unsafe_allow_html=True)
    features = [
        ("🎥", "Movie Deep-Dives",      "Ratings, budget, revenue, cast, and full breakdowns for any film."),
        ("📈", "Trending &amp; Popular", "Discover what's hot right now with real-time popularity data."),
        ("⭐", "Top Rated Films",        "Browse the highest-rated movies across genres and decades."),
        ("🎭", "Now Playing",            "See what's currently in theaters near you."),
        ("🔗", "Smart Recommendations",  "Get personalized suggestions based on movies you love."),
        ("📺", "TV Show Intelligence",   "Air dates, next episodes, and season tracking for any series."),
        ("💬", "Multi-Chat Support",     "Run parallel conversations — switch between topics effortlessly."),
    ]
    for icon, title, desc in features:
        st.markdown(
            f"<div class='feature-card'>"
            f"<div class='feature-icon'>{icon}</div>"
            f"<div class='feature-title'>{title}</div>"
            f"<div class='feature-desc'>{desc}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("<div style='height:1.5rem'></div>", unsafe_allow_html=True)

    
    _, col_cta, _ = st.columns([1, 2, 1])
    with col_cta:
        if st.button("🚀  Start a Conversation", use_container_width=True, type="primary"):
            _create_conv_with_prompt()

st.markdown(
    "<div class='app-footer'>"
    "Movie Analyst Agent v1.0 &nbsp;·&nbsp; Powered by "
    "<a href='https://concentrate.ai' target='_blank'>Concentrate AI</a>"
    "</div>",
    unsafe_allow_html=True,
)