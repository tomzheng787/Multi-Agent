"""Redis-backed persistence: conversation memory, evidence and checkpoints.

Fault tolerance:

* :func:`connect_redis` degrades to an in-process store (fakeredis) when the
  configured Redis is unreachable, so a run still completes - non-durably -
  instead of crashing.
* :class:`RedisChatMessageHistory` retries transient Redis errors and buffers
  writes it could not persist, flushing them on the next successful call.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Iterable, Sequence

import redis
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import BaseMessage, message_to_dict, messages_from_dict
from pydantic import BaseModel, Field
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

log = logging.getLogger(__name__)

_TRANSIENT = (redis.exceptions.ConnectionError, redis.exceptions.TimeoutError)
_redis_retry = retry(
    retry=retry_if_exception_type(_TRANSIENT),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=0.2, max=2),
    reraise=True,
)


def connect_redis(url: str, *, allow_fallback: bool = True) -> tuple[redis.Redis, bool]:
    """Return ``(client, durable)``.

    ``fakeredis://`` always yields an in-process store. For a real URL the
    server is pinged; if that fails and ``allow_fallback`` is set we fall back
    to an in-process store and report ``durable=False``.
    """
    import fakeredis

    if url.startswith("fakeredis://"):
        return fakeredis.FakeRedis(decode_responses=True), False
    client = redis.Redis.from_url(url, decode_responses=True, socket_timeout=5, socket_connect_timeout=3)
    try:
        client.ping()
        return client, True
    except _TRANSIENT as exc:
        if not allow_fallback:
            raise
        log.warning("Redis at %s unreachable (%s); using in-process store - state will not persist", url, exc)
        return fakeredis.FakeRedis(decode_responses=True), False


# -- conversation memory ----------------------------------------------------


class RedisChatMessageHistory(BaseChatMessageHistory):
    """LangChain chat history persisted as a capped, TTL'd Redis list."""

    def __init__(
        self,
        client: redis.Redis,
        session_id: str,
        *,
        namespace: str = "memory",
        ttl_seconds: int | None = None,
        max_messages: int | None = None,
    ):
        self.client = client
        self.key = f"{namespace}:{session_id}"
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages
        self._pending: list[str] = []  # writes not yet persisted
        self._lock = threading.Lock()

    @property
    def messages(self) -> list[BaseMessage]:  # type: ignore[override]
        try:
            self._flush_pending()
            raw = self._read()
        except _TRANSIENT as exc:
            log.warning("memory read failed for %s: %s", self.key, exc)
            raw = []
        with self._lock:
            raw = raw + list(self._pending)
        return messages_from_dict([json.loads(item) for item in raw])

    def add_messages(self, messages: Sequence[BaseMessage]) -> None:
        payload = [json.dumps(message_to_dict(m)) for m in messages]
        with self._lock:
            self._pending.extend(payload)
        try:
            self._flush_pending()
        except _TRANSIENT as exc:
            log.warning("memory write buffered for %s: %s", self.key, exc)

    def clear(self) -> None:
        with self._lock:
            self._pending.clear()
        self.client.delete(self.key)

    @property
    def pending_writes(self) -> int:
        return len(self._pending)

    @_redis_retry
    def _read(self) -> list[str]:
        return self.client.lrange(self.key, 0, -1)

    @_redis_retry
    def _flush_pending(self) -> None:
        with self._lock:
            if not self._pending:
                return
            batch = list(self._pending)
            pipe = self.client.pipeline()
            pipe.rpush(self.key, *batch)
            if self.max_messages:
                pipe.ltrim(self.key, -self.max_messages, -1)
            if self.ttl_seconds:
                pipe.expire(self.key, self.ttl_seconds)
            pipe.execute()
            del self._pending[: len(batch)]


class MemoryManager:
    """Factory for the session-level and per-agent conversation histories."""

    def __init__(self, client: redis.Redis, *, ttl_seconds: int, max_messages: int):
        self.client = client
        self.ttl_seconds = ttl_seconds
        self.max_messages = max_messages

    def session(self, session_id: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            self.client, session_id, namespace="memory:session",
            ttl_seconds=self.ttl_seconds, max_messages=self.max_messages,
        )

    def agent(self, session_id: str, role: str) -> RedisChatMessageHistory:
        return RedisChatMessageHistory(
            self.client, f"{session_id}:{role}", namespace="memory:agent",
            ttl_seconds=self.ttl_seconds, max_messages=self.max_messages,
        )

    def recall(self, session_id: str, role: str | None = None, *, limit: int = 6, max_chars: int = 1500) -> str:
        histories = [("session", self.session(session_id))]
        if role:
            histories.append((f"{role} notes", self.agent(session_id, role)))
        sections = []
        for label, history in histories:
            recent = history.messages[-limit:]
            if not recent:
                continue
            lines = [f"- [{m.type}] {_clip(str(m.content), max_chars)}" for m in recent]
            sections.append(f"{label}:\n" + "\n".join(lines))
        return "\n\n".join(sections) or "No prior memory for this session."


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


# -- evidence -----------------------------------------------------------------


class Evidence(BaseModel):
    id: str
    source: str
    content: str
    title: str | None = None
    url: str | None = None
    retrieved_at: float = Field(default_factory=time.time)

    def render(self, max_chars: int = 1200) -> str:
        where = f" <{self.url}>" if self.url else ""
        title = f" {self.title}" if self.title else ""
        return f"[{self.id}] ({self.source}){title}{where}\n{_clip(self.content, max_chars)}"


_WORD = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> set[str]:
    return {t for t in _WORD.findall(text.lower()) if len(t) > 2}


class EvidenceStore:
    """Run-scoped store of every observation gathered by tools.

    Agents cite evidence by id (``[E3]``); the verifier checks claims against
    exactly these records.
    """

    def __init__(self, client: redis.Redis, run_id: str, *, ttl_seconds: int | None = None):
        self.client = client
        self.key = f"evidence:{run_id}"
        self.seq_key = f"{self.key}:seq"
        self.dedupe_key = f"{self.key}:digest"
        self.ttl_seconds = ttl_seconds

    def add(self, *, source: str, content: str, title: str | None = None, url: str | None = None) -> Evidence:
        digest = hashlib.sha1(f"{source}|{content}".encode()).hexdigest()
        existing = self.client.hget(self.dedupe_key, digest)
        if existing:
            found = self.get(existing)
            if found:
                return found
        evidence_id = f"E{self.client.incr(self.seq_key)}"
        item = Evidence(id=evidence_id, source=source, content=content, title=title, url=url)
        pipe = self.client.pipeline()
        pipe.hset(self.key, evidence_id, item.model_dump_json())
        pipe.hset(self.dedupe_key, digest, evidence_id)
        if self.ttl_seconds:
            for key in (self.key, self.seq_key, self.dedupe_key):
                pipe.expire(key, self.ttl_seconds)
        pipe.execute()
        return item

    def get(self, evidence_id: str) -> Evidence | None:
        raw = self.client.hget(self.key, evidence_id.strip().upper())
        return Evidence.model_validate_json(raw) if raw else None

    def get_many(self, ids: Iterable[str]) -> list[Evidence]:
        return [e for e in (self.get(i) for i in ids) if e is not None]

    def exists(self, evidence_id: str) -> bool:
        return bool(self.client.hexists(self.key, evidence_id.strip().upper()))

    def all(self) -> list[Evidence]:
        items = [Evidence.model_validate_json(v) for v in self.client.hvals(self.key)]
        return sorted(items, key=lambda e: int(e.id[1:]))

    def search(self, query: str, k: int = 5) -> list[Evidence]:
        wanted = _tokens(query)
        if not wanted:
            return self.all()[:k]
        scored = []
        for item in self.all():
            overlap = len(wanted & _tokens(f"{item.title or ''} {item.content}"))
            if overlap:
                scored.append((overlap, item))
        scored.sort(key=lambda pair: (-pair[0], int(pair[1].id[1:])))
        return [item for _, item in scored[:k]]

    def render(self, items: Sequence[Evidence] | None = None, max_chars: int = 1200) -> str:
        items = self.all() if items is None else items
        return "\n\n".join(e.render(max_chars) for e in items) or "(no evidence recorded)"


# -- checkpoints --------------------------------------------------------------


class CheckpointStore:
    """Per-run task outputs, so a crashed or retried run resumes where it stopped."""

    def __init__(self, client: redis.Redis, run_id: str, *, ttl_seconds: int | None = None):
        self.client = client
        self.key = f"run:{run_id}:checkpoints"
        self.meta_key = f"run:{run_id}:meta"
        self.ttl_seconds = ttl_seconds

    def save(self, task_name: str, output: str, **meta: Any) -> None:
        record = json.dumps({"output": output, "saved_at": time.time(), **meta})
        pipe = self.client.pipeline()
        pipe.hset(self.key, task_name, record)
        if self.ttl_seconds:
            pipe.expire(self.key, self.ttl_seconds)
        pipe.execute()

    def completed(self) -> dict[str, str]:
        return {name: json.loads(raw)["output"] for name, raw in self.client.hgetall(self.key).items()}

    def set_meta(self, **fields: Any) -> None:
        self.client.hset(self.meta_key, mapping={k: json.dumps(v) for k, v in fields.items()})
        if self.ttl_seconds:
            self.client.expire(self.meta_key, self.ttl_seconds)

    def meta(self) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.client.hgetall(self.meta_key).items()}
