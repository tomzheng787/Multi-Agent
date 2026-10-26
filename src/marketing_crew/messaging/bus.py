"""Structured inter-agent messaging over Redis Streams.

Delivery is at-least-once: consumers read through a consumer group, ack on
success, are re-delivered on failure (with an attempt counter) and end up in
a dead-letter stream after ``max_attempts``. Messages left pending by a
crashed consumer are reclaimed with ``XAUTOCLAIM``.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

import redis
from pydantic import BaseModel, Field

log = logging.getLogger(__name__)


class MessageType(str, Enum):
    TASK_ASSIGNED = "task_assigned"
    TASK_COMPLETED = "task_completed"
    INFO = "info"
    FEEDBACK = "feedback"
    REVISION_REQUEST = "revision_request"
    APPROVAL = "approval"
    VERIFICATION_REPORT = "verification_report"
    ERROR = "error"
    JOB = "job"


class AgentMessage(BaseModel):
    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    run_id: str
    sender: str
    recipient: str
    type: MessageType
    payload: dict[str, Any] = Field(default_factory=dict)
    correlation_id: str | None = None
    attempt: int = 1
    created_at: float = Field(default_factory=time.time)


@dataclass(frozen=True)
class Delivery:
    entry_id: str
    message: AgentMessage


@dataclass
class ProcessStats:
    succeeded: int = 0
    retried: int = 0
    dead_lettered: int = 0


class MessageBus:
    """Redis Streams message bus, one stream per recipient."""

    EVENTS = "_events"
    DLQ = "_dlq"

    def __init__(self, client: redis.Redis, namespace: str = "bus", *, maxlen: int = 10_000, max_attempts: int = 3):
        self.client = client
        self.namespace = namespace
        self.maxlen = maxlen
        self.max_attempts = max_attempts
        self._groups: set[tuple[str, str]] = set()

    def stream(self, recipient: str) -> str:
        return f"{self.namespace}:{recipient}"

    # -- produce ------------------------------------------------------------
    def publish(self, message: AgentMessage, *, mirror_to_events: bool = True) -> str:
        data = {"data": message.model_dump_json()}
        entry_id = self.client.xadd(self.stream(message.recipient), data, maxlen=self.maxlen, approximate=True)
        if mirror_to_events and message.recipient != self.EVENTS:
            # A single firehose makes runs observable/replayable end to end.
            self.client.xadd(self.stream(self.EVENTS), data, maxlen=self.maxlen, approximate=True)
        return entry_id

    # -- consume ------------------------------------------------------------
    def ensure_group(self, recipient: str, group: str) -> None:
        if (recipient, group) in self._groups:
            return
        try:
            self.client.xgroup_create(self.stream(recipient), group, id="0", mkstream=True)
        except redis.exceptions.ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise
        self._groups.add((recipient, group))

    def consume(
        self, recipient: str, group: str, consumer: str, *, count: int = 10, block_ms: int | None = None
    ) -> list[Delivery]:
        self.ensure_group(recipient, group)
        response = self.client.xreadgroup(
            group, consumer, {self.stream(recipient): ">"}, count=count, block=block_ms
        )
        return [d for _, entries in (response or []) for d in self._decode(entries)]

    def reclaim(
        self, recipient: str, group: str, consumer: str, *, min_idle_ms: int = 60_000, count: int = 10
    ) -> list[Delivery]:
        """Take over messages a crashed consumer read but never acked."""
        self.ensure_group(recipient, group)
        result = self.client.xautoclaim(
            self.stream(recipient), group, consumer, min_idle_time=min_idle_ms, start_id="0-0", count=count
        )
        entries = result[1] if result else []
        return self._decode(entries)

    def ack(self, recipient: str, group: str, entry_id: str) -> None:
        self.client.xack(self.stream(recipient), group, entry_id)

    def fail(self, recipient: str, group: str, delivery: Delivery, error: str) -> str:
        """Re-queue a failed message, or dead-letter it once attempts run out."""
        message = delivery.message
        if message.attempt < self.max_attempts:
            retry = message.model_copy(update={"attempt": message.attempt + 1})
            self.publish(retry, mirror_to_events=False)
            outcome = "retried"
        else:
            dead = message.model_copy(update={"payload": {**message.payload, "_error": error}})
            self.client.xadd(self.stream(self.DLQ), {"data": dead.model_dump_json()}, maxlen=self.maxlen, approximate=True)
            outcome = "dead_lettered"
            log.error("message %s dead-lettered after %s attempts: %s", message.id, message.attempt, error)
        self.ack(recipient, group, delivery.entry_id)
        return outcome

    def process(
        self,
        recipient: str,
        group: str,
        consumer: str,
        handler: Callable[[AgentMessage], Any],
        *,
        count: int = 10,
        block_ms: int | None = None,
        reclaim_idle_ms: int | None = 60_000,
    ) -> ProcessStats:
        stats = ProcessStats()
        deliveries: list[Delivery] = []
        if reclaim_idle_ms is not None:
            deliveries += self.reclaim(recipient, group, consumer, min_idle_ms=reclaim_idle_ms, count=count)
        deliveries += self.consume(recipient, group, consumer, count=count, block_ms=block_ms)
        for delivery in deliveries:
            try:
                handler(delivery.message)
            except Exception as exc:  # handler failures are data, not crashes
                outcome = self.fail(recipient, group, delivery, f"{type(exc).__name__}: {exc}")
                if outcome == "retried":
                    stats.retried += 1
                else:
                    stats.dead_lettered += 1
            else:
                self.ack(recipient, group, delivery.entry_id)
                stats.succeeded += 1
        return stats

    # -- inspection ---------------------------------------------------------
    def history(self, recipient: str, count: int = 50) -> list[AgentMessage]:
        entries = self.client.xrevrange(self.stream(recipient), count=count)
        return [d.message for d in self._decode(list(reversed(entries)))]

    def dead_letters(self, count: int = 100) -> list[AgentMessage]:
        return self.history(self.DLQ, count)

    @staticmethod
    def _decode(entries: list[Any]) -> list[Delivery]:
        out = []
        for entry_id, fields in entries:
            if not fields:  # entry trimmed away while pending
                continue
            out.append(Delivery(entry_id, AgentMessage.model_validate_json(fields["data"])))
        return out
