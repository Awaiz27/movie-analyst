"""
Movie Analyst Agent — Streamlit Frontend
"""

import streamlit as st
import requests
import os
import uuid
import logging
import json
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
    """Persist conversations dict to disk."""
    try:
        payload = {
            "conversations": st.session_state.conversations,
            "active_conv": st.session_state.active_conv,
            "selected_model": st.session_state.selected_model,
            "use_streaming": st.session_state.use_streaming,
        }
        _CONV_FILE.write_text(json.dumps(payload, default=str), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to save conversations: %s", exc)


def _load_conversations() -> None:
    """Load conversations from disk into session_state (once per session)."""
    if _CONV_FILE.exists():
        try:
            data = json.loads(_CONV_FILE.read_text(encoding="utf-8"))
            st.session_state.conversations = data.get("conversations", {})
            st.session_state.active_conv = data.get("active_conv")
            st.session_state.selected_model = data.get("selected_model", "auto")
            st.session_state.use_streaming = data.get("use_streaming", True)
            # Validate active_conv still exists
            if st.session_state.active_conv not in st.session_state.conversations:
                st.session_state.active_conv = None
            return
        except Exception as exc:
            logger.warning("Failed to load conversations: %s", exc)
    # Defaults
    st.session_state.conversations = {}
    st.session_state.active_conv = None
    st.session_state.selected_model = "auto"
    st.session_state.use_streaming = True


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
        conv_id = str(uuid.uuid4())
        st.session_state.conversations[conv_id] = {
            "session_id": sid,
            "messages": [],
            "title": "New Chat",
            "created_at": datetime.now().strftime("%b %d, %H:%M"),
            "model": st.session_state.selected_model,
        }
        st.session_state.active_conv = conv_id
        if prompt_text:
            st.session_state.pending_prompt = prompt_text
        _save_conversations()
        st.rerun()


# Convenience accessor
def _active() -> Optional[dict]:
    cid = st.session_state.active_conv
    return st.session_state.conversations.get(cid) if cid else None

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
            conv_id = str(uuid.uuid4())
            st.session_state.conversations[conv_id] = {
                "session_id": sid,
                "messages": [],
                "title": "New Chat",
                "created_at": datetime.now().strftime("%b %d, %H:%M"),
                "model": st.session_state.selected_model,
            }
            st.session_state.active_conv = conv_id
            _save_conversations()
            st.rerun()

    # Conversation list
    convos = st.session_state.conversations
    if convos:
        st.markdown("<div class='sb-heading'>Conversations</div>", unsafe_allow_html=True)
        for cid in reversed(list(convos.keys())):
            c = convos[cid]
            is_active = cid == st.session_state.active_conv
            n_msgs = len([m for m in c["messages"] if m["role"] == "user"])
            title = c.get("title", "New Chat")
            if st.button(
                f"{'▸ ' if is_active else '  '}{title}  ({n_msgs})",
                key=f"conv_{cid}",
                use_container_width=True,
            ):
                st.session_state.active_conv = cid
                _save_conversations()
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

        user_input = st.chat_input("Ask about any movie, show, or cinematic topic…")
        prompt = pending or user_input

        if prompt:
            active["messages"].append({"role": "user", "content": prompt})
            if active["title"] == "New Chat":
                active["title"] = _auto_title(active["messages"])

            with st.chat_message("user", avatar="🧑‍💻"):
                st.markdown(prompt)

            with st.chat_message("assistant", avatar="🎬"):
                if st.session_state.use_streaming:
                    response = send_message(
                        active["session_id"], prompt,
                        model=st.session_state.selected_model, use_streaming=True,
                    )
                else:
                    with st.spinner("Analyzing…"):
                        response = send_message(
                            active["session_id"], prompt,
                            model=st.session_state.selected_model, use_streaming=False,
                        )
                    if response:
                        st.markdown(response)

            if response:
                active["messages"].append({"role": "assistant", "content": response})
            else:
                st.warning("No response received — check the backend logs.")
            _save_conversations()

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