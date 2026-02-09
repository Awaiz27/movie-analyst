from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    AsyncGenerator,
    AsyncIterator,
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    Iterator,
    List,
    Optional,
    TypeVar,
    Union,
    cast,
)
import json

import httpx

from llama_index.core.llms.callbacks import llm_completion_callback
from llama_index.core.llms import (
    ChatMessage,
    ChatResponse,
    CompletionResponse,
    CustomLLM,
    LLMMetadata,
)

JsonDict = Dict[str, Any]
InputType = Union[str, List[Dict[str, Any]]]
T = TypeVar("T")


class ConcentrateAPIError(RuntimeError):
    pass


@dataclass(frozen=True)
class _StopTypes:
    completed: str = "response.completed"
    failed: str = "response.failed"
    canceled: str = "response.canceled"
    incomplete: str = "response.incomplete"
    error: str = "error"


class AwaitableAsyncStream(AsyncIterator[T], Awaitable[T]):
    """
    Wrap an async generator so it is BOTH:
      - async-iterable (async for)
      - awaitable (await returns final item)

    Needed for LlamaIndex workflow agents.
    """

    def __init__(self, agen: AsyncGenerator[T, None], default_final: Callable[[], T]) -> None:
        self._agen: AsyncGenerator[T, None] = agen
        self._default_final: Callable[[], T] = default_final
        self._last: Optional[T] = None
        self._drained: bool = False

    def __aiter__(self) -> "AwaitableAsyncStream[T]":
        return self

    async def __anext__(self) -> T:
        try:
            item = await self._agen.__anext__()
        except StopAsyncIteration:
            self._drained = True
            raise
        self._last = item
        return item

    def __await__(self):
        async def _drain() -> T:
            if self._drained:
                return self._last if self._last is not None else self._default_final()

            last = self._last
            async for item in self._agen:
                last = item
            self._drained = True
            self._last = last
            return last if last is not None else self._default_final()

        return _drain().__await__()


class ConcentrateResponsesLLM(CustomLLM):
    """
    LlamaIndex CustomLLM adapter for Concentrate AI /responses endpoint.

    Important:
    - This is a pydantic-backed model (via CustomLLM), so we declare fields here and do NOT override __init__.
    """

    model: str
    api_key: str
    base_url: str = "https://api.concentrate.ai/v1"
    timeout: int = 60
    default_max_output_tokens: Optional[int] = None
    default_tool_choice: Optional[object] = "none"

    @property
    def metadata(self) -> LLMMetadata:
        return LLMMetadata(context_window=8192, num_output=1024, is_chat_model=True)

    # -----------------------------
    # Low-level helpers
    # -----------------------------
    def _url(self) -> str:
        return f"{self.base_url.rstrip('/')}/responses"

    def _headers(self) -> Dict[str, str]:
        return {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "Authorization": f"Bearer {self.api_key}",
        }

    def _messages_to_input(self, messages: Iterable[ChatMessage]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            role = getattr(m, "role", "user") or "user"
            content = getattr(m, "content", "") or ""
            out.append({"role": role, "content": content})
        return out

    def _build_payload(self, *, input_value: InputType, stream: bool, **params: Any) -> JsonDict:
        """
        Build /responses payload.

        Escape hatch:
        - raw_input: if provided, used verbatim for "input" (string OR list of objects)

        Pass-through:
        - temperature, top_p, max_output_tokens, tools, tool_choice, parallel_tool_calls, routing, etc.
        """
        raw_input = params.pop("raw_input", None)
        if raw_input is not None:
            input_value = cast(InputType, raw_input)

        payload: JsonDict = {"model": self.model, "input": input_value, "stream": stream}

        if self.default_max_output_tokens is not None and "max_output_tokens" not in params:
            payload["max_output_tokens"] = self.default_max_output_tokens

        # apply default tool_choice unless caller overrides
        if "tool_choice" not in params and self.default_tool_choice is not None:
            payload["tool_choice"] = self.default_tool_choice
        else:
            payload["tool_choice"] = "none"
        
        for k, v in params.items():
            if v is None:
                continue
            if k in {"prompt", "messages"}:
                continue
            payload[k] = v

        return payload

    # -----------------------------
    # JSON output extraction
    # -----------------------------
    def _extract_text(self, data: Any) -> str:
        """
        Extract assistant text from /responses JSON.

        Common Concentrate shapes include:
        - {"output_text": "..."}
        - {"output":[{"type":"message","role":"assistant","content":[{"type":"output_text","text":"..."}]}]}
        """
        if isinstance(data, dict):
            out_text = data.get("output_text")
            if isinstance(out_text, str) and out_text.strip():
                return out_text

            output = data.get("output")
            if isinstance(output, list):
                parts: List[str] = []
                for item in output:
                    if not isinstance(item, dict):
                        continue
                    if item.get("type") == "message" and item.get("role") == "assistant":
                        content = item.get("content")
                        if isinstance(content, list):
                            for block in content:
                                if isinstance(block, dict):
                                    txt = block.get("text")
                                    if isinstance(txt, str) and txt:
                                        parts.append(txt)
                if parts:
                    return "".join(parts)

            txt = data.get("text")
            if isinstance(txt, str) and txt.strip():
                return txt

        return str(data)

    # -----------------------------
    # Non-stream requests
    # -----------------------------
    def _post_json(self, payload: JsonDict) -> JsonDict:
        with httpx.Client(timeout=self.timeout) as client:
            resp = client.post(self._url(), json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict):
            raise ConcentrateAPIError("Unexpected /responses response (expected JSON object).")
        return cast(JsonDict, data)

    async def _apost_json(self, payload: JsonDict) -> JsonDict:
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.post(self._url(), json=payload, headers=self._headers())
            resp.raise_for_status()
            data = resp.json()
        if not isinstance(data, dict):
            raise ConcentrateAPIError("Unexpected /responses response (expected JSON object).")
        return cast(JsonDict, data)

    # -----------------------------
    # SSE parsing
    # -----------------------------
    def _iter_sse_events(self, resp: httpx.Response) -> Iterator[JsonDict]:
        """
        SSE is blocks separated by blank lines. Each block can have multiple data: lines.
        """
        buf: List[str] = []
        for raw in resp.iter_lines():
            if raw is None:
                continue
            line = raw.decode("utf-8", errors="replace") if isinstance(raw, (bytes, bytearray)) else str(raw)
            line = line.rstrip("\r\n")

            if line == "":
                if not buf:
                    continue
                data_lines = [l[5:].lstrip() for l in buf if l.startswith("data:")]
                buf = []
                if not data_lines:
                    continue
                try:
                    evt = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    continue
                if isinstance(evt, dict):
                    yield cast(JsonDict, evt)
                continue

            buf.append(line)

        # flush tail
        if buf:
            data_lines = [l[5:].lstrip() for l in buf if l.startswith("data:")]
            if data_lines:
                try:
                    evt = json.loads("\n".join(data_lines))
                    if isinstance(evt, dict):
                        yield cast(JsonDict, evt)
                except json.JSONDecodeError:
                    pass

    async def _aiter_sse_events(self, resp: httpx.Response) -> AsyncIterator[JsonDict]:
        buf: List[str] = []
        async for line in resp.aiter_lines():
            if line is None:
                continue
            s = str(line).rstrip("\r\n")

            if s == "":
                if not buf:
                    continue
                data_lines = [l[5:].lstrip() for l in buf if l.startswith("data:")]
                buf = []
                if not data_lines:
                    continue
                try:
                    evt = json.loads("\n".join(data_lines))
                except json.JSONDecodeError:
                    continue
                if isinstance(evt, dict):
                    yield cast(JsonDict, evt)
                continue

            buf.append(s)

        if buf:
            data_lines = [l[5:].lstrip() for l in buf if l.startswith("data:")]
            if data_lines:
                try:
                    evt = json.loads("\n".join(data_lines))
                    if isinstance(evt, dict):
                        yield cast(JsonDict, evt)
                except json.JSONDecodeError:
                    pass

    # -----------------------------
    # Streaming text extraction (robust)
    # -----------------------------
    def _iter_stream_text(self, payload: JsonDict) -> Generator[str, None, None]:
        stop = _StopTypes()
        saw_any_text = False
        saw_delta = False

        with httpx.Client(timeout=self.timeout) as client:
            with client.stream("POST", self._url(), json=payload, headers=self._headers()) as resp:
                resp.raise_for_status()
                for evt in self._iter_sse_events(resp):
                    etype = evt.get("type")

                    if etype == "response.output_text.delta":
                        delta = evt.get("delta")
                        if isinstance(delta, str) and delta:
                            saw_any_text = True
                            saw_delta = True
                            yield delta

                    elif etype == "response.output_text.done":
                        text = evt.get("text")
                        if isinstance(text, str) and text and not saw_delta:
                            # Some models/vendors only provide text at "done"
                            saw_any_text = True
                            yield text

                    elif etype == "response.content_part.added":
                        # Optional: some backends include text here
                        part = evt.get("part")
                        if isinstance(part, dict):
                            t = part.get("text")
                            if isinstance(t, str) and t:
                                saw_any_text = True
                                yield t

                    elif etype in (stop.completed, stop.failed, stop.canceled, stop.incomplete, stop.error):
                        break

        # Final fallback: if stream yielded nothing, make a non-stream call once
        if not saw_any_text:
            fallback_payload = dict(payload)
            fallback_payload["stream"] = False
            data = self._post_json(fallback_payload)
            text = self._extract_text(data)
            if text:
                yield text

    async def _aiter_stream_text(self, payload: JsonDict) -> AsyncGenerator[str, None]:
        stop = _StopTypes()
        saw_any_text = False
        saw_delta = False

        async with httpx.AsyncClient(timeout=self.timeout) as client:
            async with client.stream("POST", self._url(), json=payload, headers=self._headers()) as resp:
                resp.raise_for_status()
                async for evt in self._aiter_sse_events(resp):
                    etype = evt.get("type")

                    if etype == "response.output_text.delta":
                        delta = evt.get("delta")
                        if isinstance(delta, str) and delta:
                            saw_any_text = True
                            saw_delta = True
                            yield delta

                    elif etype == "response.output_text.done":
                        text = evt.get("text")
                        if isinstance(text, str) and text and not saw_delta:
                            saw_any_text = True
                            yield text

                    elif etype == "response.content_part.added":
                        part = evt.get("part")
                        if isinstance(part, dict):
                            t = part.get("text")
                            if isinstance(t, str) and t:
                                saw_any_text = True
                                yield t

                    elif etype in (stop.completed, stop.failed, stop.canceled, stop.incomplete, stop.error):
                        break

        if not saw_any_text:
            fallback_payload = dict(payload)
            fallback_payload["stream"] = False
            data = await self._apost_json(fallback_payload)
            text = self._extract_text(data)
            if text:
                yield text

    # -----------------------------
    # LlamaIndex Completion API
    # -----------------------------
    @llm_completion_callback()
    def complete(self, prompt: str, **params: Any) -> CompletionResponse:
        payload = self._build_payload(input_value=prompt, stream=False, **params)
        data = self._post_json(payload)
        return CompletionResponse(text=self._extract_text(data))

    @llm_completion_callback()
    async def acomplete(self, prompt: str, **params: Any) -> CompletionResponse:
        payload = self._build_payload(input_value=prompt, stream=False, **params)
        data = await self._apost_json(payload)
        return CompletionResponse(text=self._extract_text(data))

    @llm_completion_callback()
    def stream_complete(self, prompt: str, **params: Any) -> Generator[CompletionResponse, None, None]:
        payload = self._build_payload(input_value=prompt, stream=True, **params)
        acc = ""
        for chunk in self._iter_stream_text(payload):
            acc += chunk
            yield CompletionResponse(text=acc, delta=chunk)

    def _astream_complete_gen(self, prompt: str, **params: Any) -> AsyncGenerator[CompletionResponse, None]:
        async def _gen() -> AsyncGenerator[CompletionResponse, None]:
            payload = self._build_payload(input_value=prompt, stream=True, **params)
            acc = ""
            async for chunk in self._aiter_stream_text(payload):
                acc += chunk
                yield CompletionResponse(text=acc, delta=chunk)

        return _gen()

    @llm_completion_callback()
    async def astream_complete(self, prompt: str, **params: Any) -> AwaitableAsyncStream[CompletionResponse]:
        def _default_final() -> CompletionResponse:
            return CompletionResponse(text="")

        return AwaitableAsyncStream(self._astream_complete_gen(prompt, **params), _default_final)

    # -----------------------------
    # LlamaIndex Chat API
    # -----------------------------
    def chat(self, messages: List[ChatMessage], **params: Any) -> ChatResponse:
        input_value = self._messages_to_input(messages)
        payload = self._build_payload(input_value=input_value, stream=False, **params)
        data = self._post_json(payload)
        text = self._extract_text(data)
        return ChatResponse(message=ChatMessage(role="assistant", content=text))

    async def achat(self, messages: List[ChatMessage], **params: Any) -> ChatResponse:
        input_value = self._messages_to_input(messages)
        payload = self._build_payload(input_value=input_value, stream=False, **params)
        data = await self._apost_json(payload)
        text = self._extract_text(data)
        return ChatResponse(message=ChatMessage(role="assistant", content=text))

    def stream_chat(self, messages: List[ChatMessage], **params: Any) -> Generator[ChatResponse, None, None]:
        input_value = self._messages_to_input(messages)
        payload = self._build_payload(input_value=input_value, stream=True, **params)
        acc = ""
        for chunk in self._iter_stream_text(payload):
            acc += chunk
            yield ChatResponse(message=ChatMessage(role="assistant", content=acc), delta=chunk)

    def _astream_chat_gen(self, messages: List[ChatMessage], **params: Any) -> AsyncGenerator[ChatResponse, None]:
        async def _gen() -> AsyncGenerator[ChatResponse, None]:
            input_value = self._messages_to_input(messages)
            payload = self._build_payload(input_value=input_value, stream=True, **params)
            acc = ""
            async for chunk in self._aiter_stream_text(payload):
                acc += chunk
                yield ChatResponse(message=ChatMessage(role="assistant", content=acc), delta=chunk)

        return _gen()

    async def astream_chat(self, messages: List[ChatMessage], **params: Any) -> AwaitableAsyncStream[ChatResponse]:
        """
        Workflow-agent compatible async streaming.
        """
        def _default_final() -> ChatResponse:
            return ChatResponse(message=ChatMessage(role="assistant", content=""))

        return AwaitableAsyncStream(self._astream_chat_gen(messages, **params), _default_final)
