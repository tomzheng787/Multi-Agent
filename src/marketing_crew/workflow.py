"""Fault-tolerant execution of the marketing crew.

* :class:`MarketingWorkflow` runs one brief end to end. Each finished task is
  checkpointed in Redis, so a crashed or retried run resumes from the last
  completed task instead of starting over. Session memory carries context
  between runs.
* :class:`JobQueue` is a durable work queue on Redis Streams: workers pick up
  briefs through a consumer group, failed jobs are retried (resuming from
  checkpoints) and dead-lettered after the attempt budget, and jobs orphaned
  by a crashed worker are reclaimed.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Callable

import anthropic
import httpx
import redis
from crewai.hooks import register_before_tool_call_hook, unregister_before_tool_call_hook
from pydantic import BaseModel, Field

from marketing_crew.crew.builder import build_crew, make_delegation_guard
from marketing_crew.messaging.bus import AgentMessage, MessageBus, MessageType
from marketing_crew.runtime import Services, build_services
from marketing_crew.settings import Settings

log = logging.getLogger(__name__)


class CampaignBrief(BaseModel):
    brand: str
    product: str
    location: str
    audience: str
    goals: str
    budget_usd: float = Field(gt=0)
    horizon_days: int = Field(default=90, gt=0)
    session_id: str = "default"

    def inputs(self) -> dict[str, str]:
        return {
            "brand": self.brand,
            "product": self.product,
            "location": self.location,
            "audience": self.audience,
            "goals": self.goals,
            "budget_usd": f"{self.budget_usd:,.0f}",
            "horizon_days": str(self.horizon_days),
        }

    def facts(self) -> str:
        """Everything the brief itself establishes; claims may restate it without citation."""
        return (
            f"budget {self.budget_usd:.0f} ({self.budget_usd:,.0f} USD); horizon {self.horizon_days} days; "
            f"location {self.location}; product {self.product}; audience {self.audience}; goals {self.goals}"
        )


@dataclass
class RunResult:
    run_id: str
    status: str
    final_output: str
    task_outputs: dict[str, str]
    resumed_tasks: list[str]
    verification: dict[str, Any]
    usage: dict[str, int]
    rbac_denials: list[dict[str, Any]]
    artifacts: list[str]
    durable_memory: bool
    duration_s: float
    search_runs: list[dict[str, Any]] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2, default=str)


def default_llm_factory(services: Services) -> Callable[[str, str | None], Any]:
    settings = services.settings
    if settings.offline:
        from marketing_crew.llm.offline import ScriptedCrewLLM

        return lambda role, effort: ScriptedCrewLLM(services=services, rbac_role=role)

    from marketing_crew.llm.crewai_adapter import ClaudeCrewLLM

    def factory(role: str, effort: str | None) -> ClaudeCrewLLM:
        default = settings.manager_effort if role == "manager" else settings.effort
        return ClaudeCrewLLM(services.llm, effort=effort or default)

    return factory


class MarketingWorkflow:
    def __init__(
        self,
        settings: Settings,
        *,
        redis_client: redis.Redis | None = None,
        anthropic_client: anthropic.Anthropic | None = None,
        http_client: httpx.Client | None = None,
    ):
        self.settings = settings
        self.redis_client = redis_client
        self.anthropic_client = anthropic_client
        self.http_client = http_client

    def services_for(self, run_id: str, session_id: str) -> Services:
        return build_services(
            self.settings, run_id=run_id, session_id=session_id, redis_client=self.redis_client,
            anthropic_client=self.anthropic_client, http_client=self.http_client,
        )

    def run(self, brief: CampaignBrief, *, run_id: str | None = None, resume: bool = True, verbose: bool = False) -> RunResult:
        started = time.monotonic()
        run_id = run_id or f"run-{uuid.uuid4().hex[:10]}"
        services = self.services_for(run_id, brief.session_id)
        services.brief = brief.inputs()
        services.brief_facts = brief.facts()
        # The Redis client may have been created by build_services (incl. fallback); reuse it.
        self.redis_client = self.redis_client or services.redis

        completed = services.checkpoints.completed() if resume else {}
        services.checkpoints.set_meta(status="running", brief=brief.model_dump(), started_at=time.time())
        inputs = {**brief.inputs(), "history": services.memory.recall(brief.session_id)}

        plan = build_crew(
            services, inputs, completed=completed,
            llm_factory=default_llm_factory(services), verbose=verbose,
        )
        if completed:
            log.info("resuming %s: skipping checkpointed tasks %s", run_id, plan.skipped)

        guard = make_delegation_guard(services.access, plan.role_by_display)
        register_before_tool_call_hook(guard)
        search_runs: list[dict[str, Any]] = []
        original_search = services.search_agent.run

        def traced_search(question: str):  # keep search traces for the run report
            result = original_search(question)
            search_runs.append({"question": question, "status": result.status, "steps": result.steps,
                                "self_corrections": result.self_corrections, "trace": result.trace_dicts()})
            return result

        services.search_agent.run = traced_search  # type: ignore[method-assign]
        try:
            if plan.crew is not None:
                plan.crew.kickoff(inputs=plan.inputs)
        except Exception as exc:
            services.checkpoints.set_meta(status="failed", error=f"{type(exc).__name__}: {exc}")
            services.publish("workflow", MessageBus.EVENTS, MessageType.ERROR, {"error": str(exc)})
            raise
        finally:
            unregister_before_tool_call_hook(guard)
            services.search_agent.run = original_search  # type: ignore[method-assign]

        outputs = services.checkpoints.completed()
        task_order = list(plan.owners)
        final_output = next((outputs[t] for t in reversed(task_order) if t in outputs), "")
        strategy = outputs.get("strategy_review") or outputs.get("strategy_draft") or final_output

        session = services.memory.session(brief.session_id)
        session.add_user_message(f"Brief ({run_id}): {brief.model_dump_json()}")
        session.add_ai_message(f"Approved strategy ({run_id}):\n{strategy[:6000]}")

        report_path = services.output_dir / "final_output.md"
        report_path.write_text(final_output, encoding="utf-8")
        services.artifacts.add(str(report_path))

        result = RunResult(
            run_id=run_id,
            status="completed",
            final_output=final_output,
            task_outputs={t: outputs[t] for t in task_order if t in outputs},
            resumed_tasks=plan.skipped,
            verification=services.ledger.summary(),
            usage=services.llm.usage.as_dict(),
            rbac_denials=[
                {"role": e.role, "permission": e.permission, "resource": e.resource}
                for e in services.access.audit_log if not e.allowed
            ],
            artifacts=sorted(services.artifacts),
            durable_memory=services.durable,
            duration_s=round(time.monotonic() - started, 2),
            search_runs=search_runs,
        )
        (services.output_dir / "run_report.json").write_text(result.to_json(), encoding="utf-8")
        services.checkpoints.set_meta(status="completed", finished_at=time.time())
        services.publish("workflow", MessageBus.EVENTS, MessageType.INFO, {"run": "completed", "run_id": run_id})
        return result


class JobQueue:
    RECIPIENT = "marketing"
    GROUP = "workers"

    def __init__(self, client: redis.Redis, *, max_attempts: int = 3):
        self.client = client
        self.bus = MessageBus(client, namespace="jobs", max_attempts=max_attempts)

    def submit(self, brief: CampaignBrief, *, job_id: str | None = None) -> str:
        job_id = job_id or f"run-{uuid.uuid4().hex[:10]}"
        self.bus.publish(
            AgentMessage(run_id=job_id, sender="client", recipient=self.RECIPIENT, type=MessageType.JOB,
                         payload=brief.model_dump()),
        )
        self._status(job_id, state="queued", attempts=0)
        return job_id

    def work(
        self,
        workflow: MarketingWorkflow,
        *,
        consumer: str,
        once: bool = False,
        block_ms: int = 5000,
        reclaim_idle_ms: int = 15 * 60 * 1000,
        on_result: Callable[[RunResult], None] | None = None,
    ) -> None:
        def handle(message: AgentMessage) -> None:
            brief = CampaignBrief.model_validate(message.payload)
            self._status(message.run_id, state="running", attempts=message.attempt, worker=consumer)
            try:
                # Retries resume from checkpoints, so finished tasks are not redone.
                result = workflow.run(brief, run_id=message.run_id, resume=True)
            except Exception as exc:
                final = message.attempt >= self.bus.max_attempts
                self._status(message.run_id, state="dead_lettered" if final else "retrying",
                             attempts=message.attempt, error=f"{type(exc).__name__}: {exc}")
                raise
            self._status(message.run_id, state="completed", attempts=message.attempt,
                         output_dir=str(workflow.settings.output_dir / message.run_id))
            if on_result:
                on_result(result)

        while True:
            stats = self.bus.process(
                self.RECIPIENT, self.GROUP, consumer, handle,
                count=1, block_ms=None if once else block_ms, reclaim_idle_ms=reclaim_idle_ms,
            )
            if once and not (stats.retried):
                return

    def status(self, job_id: str) -> dict[str, Any]:
        return {k: json.loads(v) for k, v in self.client.hgetall(f"job:{job_id}").items()}

    def dead_letters(self) -> list[AgentMessage]:
        return self.bus.dead_letters()

    def _status(self, job_id: str, **fields: Any) -> None:
        self.client.hset(f"job:{job_id}", mapping={k: json.dumps(v) for k, v in {**fields, "updated_at": time.time()}.items()})
