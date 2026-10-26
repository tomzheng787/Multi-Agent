"""Service container wiring one run's infrastructure together."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import anthropic
import httpx
import redis

from marketing_crew.llm.claude import ClaudeClient
from marketing_crew.memory.store import CheckpointStore, EvidenceStore, MemoryManager, connect_redis
from marketing_crew.messaging.bus import AgentMessage, MessageBus, MessageType
from marketing_crew.security.rbac import AccessController, AuditEvent
from marketing_crew.settings import Settings
from marketing_crew.tools.api_registry import ApiInvoker, ApiRegistry, HttpFetcher
from marketing_crew.tools.local_search_agent import LocalSearchAgent
from marketing_crew.verification.cove import ChainOfVerification, VerificationLedger


@dataclass
class Services:
    settings: Settings
    run_id: str
    session_id: str
    redis: redis.Redis
    durable: bool
    access: AccessController
    memory: MemoryManager
    evidence: EvidenceStore
    checkpoints: CheckpointStore
    bus: MessageBus
    llm: ClaudeClient
    invoker: ApiInvoker
    search_agent: LocalSearchAgent
    verifier: ChainOfVerification
    ledger: VerificationLedger
    output_dir: Path
    brief: dict[str, str] = field(default_factory=dict)
    brief_facts: str = ""
    task_owner_roles: dict[str, str] = field(default_factory=dict)  # task name -> agent display role
    artifacts: set[str] = field(default_factory=set)

    def publish(self, sender: str, recipient: str, type_: MessageType, payload: dict[str, Any]) -> str:
        message = AgentMessage(run_id=self.run_id, sender=sender, recipient=recipient, type=type_, payload=payload)
        self.bus.publish(message)
        return message.id


def build_services(
    settings: Settings,
    *,
    run_id: str,
    session_id: str = "default",
    redis_client: redis.Redis | None = None,
    anthropic_client: anthropic.Anthropic | None = None,
    http_client: httpx.Client | None = None,
) -> Services:
    if redis_client is not None:
        client, durable = redis_client, "fakeredis" not in type(redis_client).__module__
    else:
        client, durable = connect_redis(settings.redis_url, allow_fallback=settings.redis_fallback)
    ttl = settings.memory_ttl_seconds

    bus = MessageBus(client, namespace=f"bus:{run_id}", max_attempts=settings.job_max_attempts)

    def audit_to_bus(event: AuditEvent) -> None:
        if not event.allowed:  # denials are what operators need to see
            bus.publish(AgentMessage(
                run_id=run_id, sender="rbac", recipient=MessageBus.EVENTS, type=MessageType.ERROR,
                payload={"denied": {"role": event.role, "permission": event.permission, "resource": event.resource}},
            ))

    access = AccessController.from_yaml(settings.config_dir / "rbac.yaml", audit_sink=audit_to_bus)
    evidence = EvidenceStore(client, run_id, ttl_seconds=ttl)

    if settings.offline:
        from marketing_crew.llm.offline import OfflineClaudeClient, offline_http_transport

        llm: ClaudeClient = OfflineClaudeClient(settings)
        http_client = http_client or httpx.Client(transport=offline_http_transport())
        verifier_llm = None  # deterministic + lexical verification offline
    else:
        llm = ClaudeClient(settings, client=anthropic_client)
        verifier_llm = llm

    registry = ApiRegistry.from_yaml(settings.config_dir / "apis.yaml")
    fetcher = HttpFetcher(http_client, user_agent=settings.http_user_agent)
    invoker = ApiInvoker(registry, fetcher)

    output_dir = settings.output_dir / run_id
    output_dir.mkdir(parents=True, exist_ok=True)

    return Services(
        settings=settings,
        run_id=run_id,
        session_id=session_id,
        redis=client,
        durable=durable,
        access=access,
        memory=MemoryManager(client, ttl_seconds=ttl, max_messages=settings.memory_max_messages),
        evidence=evidence,
        checkpoints=CheckpointStore(client, run_id, ttl_seconds=ttl),
        bus=bus,
        llm=llm,
        invoker=invoker,
        search_agent=LocalSearchAgent(llm, invoker, evidence, max_steps=settings.search_max_steps),
        verifier=ChainOfVerification(evidence, verifier_llm, effort=settings.verifier_effort),
        ledger=VerificationLedger(),
        output_dir=output_dir,
    )
