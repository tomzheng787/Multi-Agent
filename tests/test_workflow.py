import json
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

from marketing_crew.crew.builder import make_delegation_guard
from marketing_crew.llm.offline import ScriptedCrewLLM
from marketing_crew.messaging.bus import MessageBus, MessageType
from marketing_crew.runtime import build_services
from marketing_crew.security.rbac import AccessController
from marketing_crew.settings import CONFIG_DIR, PROJECT_ROOT
from marketing_crew.tools.toolkit import (
    ChannelWeight,
    allocate_budget,
    build_agent_tools,
    build_langchain_tools,
    safe_artifact_path,
    to_crewai_tool,
)
from marketing_crew.workflow import CampaignBrief, JobQueue, MarketingWorkflow

TASKS = ["market_research", "market_analysis", "strategy_draft", "strategy_review", "execution_plan"]


@pytest.fixture
def brief():
    return CampaignBrief.model_validate_json((PROJECT_ROOT / "examples" / "brief.json").read_text())


@contextmanager
def critic_crashes():
    original = ScriptedCrewLLM._critic

    def crash(self, **_):
        raise RuntimeError("simulated outage")

    ScriptedCrewLLM._critic = crash
    try:
        yield
    finally:
        ScriptedCrewLLM._critic = original


def test_offline_end_to_end(settings, redis_client, brief):
    result = MarketingWorkflow(settings, redis_client=redis_client).run(brief, run_id="r1")

    assert list(result.task_outputs) == TASKS
    verification = result.verification["tasks"]
    assert verification["strategy_draft"]["attempts"] == 2  # fabricated stat rejected, then fixed
    assert verification["strategy_draft"]["first_rate"] > settings.max_hallucination_rate
    assert all(t["final_rate"] == 0.0 and t["passed"] for t in verification.values())
    assert "87%" not in result.task_outputs["strategy_review"]
    assert {Path(a).name for a in result.artifacts} >= {"strategy.md", "launch_plan.md", "campaign_schedule.json"}
    assert result.search_runs[0]["self_corrections"] == 1
    assert result.rbac_denials == []

    bus = MessageBus(redis_client, "bus:r1")
    events = bus.history(MessageBus.EVENTS, count=500)
    completed = [e.payload["task"] for e in events if e.type == MessageType.TASK_COMPLETED]
    assert completed == TASKS
    assert any(e.type == MessageType.APPROVAL and e.recipient == "executor" for e in events)

    memory = MarketingWorkflow(settings, redis_client=redis_client).services_for("r2", brief.session_id).memory
    assert "Approved strategy (r1)" in memory.recall(brief.session_id)


def test_crash_then_resume_from_checkpoints(settings, redis_client, brief):
    workflow = MarketingWorkflow(settings, redis_client=redis_client)
    with critic_crashes(), pytest.raises(Exception):
        workflow.run(brief, run_id="r-crash")
    services = workflow.services_for("r-crash", brief.session_id)
    assert list(services.checkpoints.completed()) == TASKS[:3]
    assert services.checkpoints.meta()["status"] == "failed"

    result = workflow.run(brief, run_id="r-crash")
    assert result.resumed_tasks == TASKS[:3]
    assert result.search_runs == []  # research was not repeated
    assert list(result.task_outputs) == TASKS


class FlakyWorkflow(MarketingWorkflow):
    def __init__(self, *args, failures: int, **kwargs):
        super().__init__(*args, **kwargs)
        self.failures = failures

    def run(self, *args, **kwargs):
        if self.failures > 0:
            self.failures -= 1
            with critic_crashes():
                return super().run(*args, **kwargs)
        return super().run(*args, **kwargs)


def test_job_queue_retries_and_resumes(settings, redis_client, brief):
    queue = JobQueue(redis_client, max_attempts=3)
    job_id = queue.submit(brief)
    queue.work(FlakyWorkflow(settings, redis_client=redis_client, failures=1), consumer="w1", once=True)
    status = queue.status(job_id)
    assert status["state"] == "completed" and status["attempts"] == 2
    assert queue.dead_letters() == []


def test_job_queue_dead_letters_after_budget(settings, redis_client, brief):
    queue = JobQueue(redis_client, max_attempts=2)
    job_id = queue.submit(brief)
    queue.work(FlakyWorkflow(settings, redis_client=redis_client, failures=5), consumer="w1", once=True)
    assert queue.status(job_id)["state"] == "dead_lettered"
    dead = queue.dead_letters()
    assert len(dead) == 1 and "did not produce a deliverable" in dead[0].payload["_error"]


def test_delegation_guard_enforces_rbac():
    access = AccessController.from_yaml(CONFIG_DIR / "rbac.yaml")
    roles = {"chief marketing officer": "manager", "strategy writer": "writer", "campaign executor": "executor"}
    guard = make_delegation_guard(access, roles)

    def ctx(agent_role, coworker, tool="Delegate work to coworker"):
        return SimpleNamespace(tool_name=tool, tool_input={"coworker": coworker}, agent=SimpleNamespace(role=agent_role))

    assert guard(ctx("Chief Marketing Officer", "Campaign Executor")) is None
    assert guard(ctx("Strategy Writer", "campaign executor")) is False
    assert guard(ctx("Chief Marketing Officer", "Chief Marketing Officer")) is False
    assert guard(ctx("Strategy Writer", "anyone", tool="evidence_lookup")) is None


def test_tools_are_least_privilege_and_guarded(settings, redis_client):
    services = build_services(settings, run_id="t", redis_client=redis_client)
    granted = build_agent_tools(services, "writer", ["evidence_lookup", "save_artifact"])
    assert [t.name for t in granted] == ["evidence_lookup"]

    leaked = to_crewai_tool(build_langchain_tools(services, "writer")["save_artifact"], role="writer", access=services.access)
    assert leaked._run(filename="x.md", content="y").startswith("ACCESS DENIED")
    assert not (services.output_dir / "x.md").exists()


def test_budget_allocation_is_exact():
    rows = allocate_budget(25_000, [ChannelWeight(name="a", weight=1), ChannelWeight(name="b", weight=2),
                                    ChannelWeight(name="c", weight=3, min_amount=1000)])
    assert sum(r["amount_usd"] for r in rows) == 25_000
    with pytest.raises(ValueError):
        allocate_budget(100, [ChannelWeight(name="a", weight=1, min_amount=500)])


def test_artifact_paths_cannot_escape(tmp_path):
    assert safe_artifact_path(tmp_path, "../../plan.md") == tmp_path / "plan.md"
    for bad in ["../../etc/passwd", "run.sh", "a b.md"]:
        with pytest.raises(ValueError):
            safe_artifact_path(tmp_path, bad)
