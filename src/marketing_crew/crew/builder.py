"""Builds the hierarchical marketing crew from YAML config, RBAC and services."""

from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable

import yaml
from crewai import Agent, BaseLLM, Crew, Process, Task

from marketing_crew.messaging.bus import MessageType
from marketing_crew.runtime import Services
from marketing_crew.security.rbac import AccessController, PermissionDenied
from marketing_crew.tools.toolkit import build_agent_tools
from marketing_crew.verification.cove import VerificationGuardrail

log = logging.getLogger(__name__)

# Checkpointing is handled by our Redis layer, not CrewAI's serialiser.
warnings.filterwarnings("ignore", message="function callbacks cannot be serialized")

DELEGATION_TOOLS = {"delegate_work_to_coworker", "ask_question_to_coworker"}

LLMFactory = Callable[[str, "str | None"], BaseLLM]


def norm_role(text: str) -> str:
    return " ".join(text.strip().strip("'\"").lower().split())


@dataclass
class CrewPlan:
    crew: Crew | None
    inputs: dict[str, str]
    remaining: list[str]
    skipped: list[str]
    owners: dict[str, str]
    role_by_display: dict[str, str] = field(default_factory=dict)


_FAILED_OUTPUT = re.compile(
    r"^(error executing (task|tool)|agent stopped due to (iteration|time) limit|tool usage failed)", re.I
)


class TaskGuardrail:
    """Output gate for every task: integrity first, then chain-of-verification.

    * Integrity failures (an empty answer, or a coworker's crash surfacing as
      text through the delegation tool) are never waived: once retries run out
      CrewAI raises, the run is marked failed and resumes from checkpoints.
    * Verification failures are retried with targeted feedback. On the final
      attempt the output is accepted with the unresolved findings appended as a
      visible warning (the ledger records it as not passed), so one stubborn
      claim degrades the deliverable explicitly instead of aborting the run.
    """

    def __init__(self, task_name: str, *, verification: VerificationGuardrail | None, max_attempts: int, min_chars: int = 80):
        self.task_name = task_name
        self.verification = verification
        self.max_attempts = max_attempts
        self.min_chars = min_chars
        self.attempts = 0

    def __call__(self, output):
        self.attempts += 1
        text = (getattr(output, "raw", None) or str(output)).strip()
        if len(text) < self.min_chars or _FAILED_OUTPUT.search(text):
            return False, (
                f"The delegated work did not produce a deliverable (got: {text[:200]!r}). "
                "Delegate the task again with the full context."
            )
        if self.verification is None:
            return True, text
        ok, value = self.verification(output)
        if ok or self.attempts < self.max_attempts:
            return ok, value
        log.warning("task %s kept unverified claims after %s attempts", self.task_name, self.attempts)
        return True, f"{text}\n\n> WARNING - unresolved verification findings:\n> " + value.replace("\n", "\n> ")


def _as_function(guardrail: TaskGuardrail) -> Callable[[Any], Any]:
    # CrewAI introspects guardrails with inspect.getsource, which needs a function.
    def task_guardrail(output):
        return guardrail(output)

    return task_guardrail


def make_delegation_guard(access: AccessController, role_by_display: dict[str, str]) -> Callable[[Any], bool | None]:
    """before_tool_call hook enforcing RBAC on CrewAI's delegation tools."""

    def guard(context: Any) -> bool | None:
        if norm_role(context.tool_name).replace(" ", "_") not in DELEGATION_TOOLS:
            return None
        caller = role_by_display.get(norm_role(getattr(context.agent, "role", "") or ""), "unknown")
        coworker = str((context.tool_input or {}).get("coworker", ""))
        target = role_by_display.get(norm_role(coworker), coworker)
        try:
            access.check_delegation(caller, target)
        except PermissionDenied as exc:
            log.warning("blocked delegation: %s", exc)
            return False
        return None

    return guard


def _load(services: Services, name: str) -> Any:
    return yaml.safe_load((services.settings.config_dir / name).read_text())


def build_crew(
    services: Services,
    brief_inputs: dict[str, str],
    *,
    completed: dict[str, str],
    llm_factory: LLMFactory,
    verbose: bool = False,
) -> CrewPlan:
    agents_cfg = _load(services, "agents.yaml")
    tasks_cfg = _load(services, "tasks.yaml")["tasks"]
    settings = services.settings

    role_by_display: dict[str, str] = {}
    workers: dict[str, Agent] = {}
    for key, spec in agents_cfg["agents"].items():
        workers[key] = Agent(
            role=spec["role"],
            goal=spec["goal"],
            backstory=spec["backstory"],
            tools=build_agent_tools(services, key, spec.get("tools", [])),
            llm=llm_factory(key, spec.get("effort")),
            allow_delegation=False,
            max_iter=15,
            verbose=verbose,
        )
        role_by_display[norm_role(spec["role"])] = key

    manager_spec = agents_cfg["manager"]
    manager_key = manager_spec["rbac_role"]
    role_by_display[norm_role(manager_spec["role"])] = manager_key

    inputs = dict(brief_inputs)
    tasks: dict[str, Task] = {}
    owners: dict[str, str] = {}
    skipped: list[str] = []
    for spec in tasks_cfg:
        name, owner = spec["name"], spec["owner"]
        owners[name] = owner
        if name in completed:
            skipped.append(name)
            continue
        # Fail at build time if the plan itself violates the delegation policy.
        if not services.access.can_delegate(manager_key, owner):
            raise PermissionDenied(manager_key, "delegate", owner)

        description = spec["description"]
        context_names = spec.get("context", [])
        checkpointed = [c for c in context_names if c in completed]
        if checkpointed:
            key = f"checkpoint_{name}"
            inputs[key] = "\n\n".join(f"### {c}\n{completed[c]}" for c in checkpointed)
            description += f"\n\nOutputs completed in an earlier attempt (checkpointed):\n{{{key}}}"

        verification = None
        if spec.get("verify"):
            verification = VerificationGuardrail(
                services.verifier,
                task_name=name,
                threshold=settings.max_hallucination_rate,
                ledger=services.ledger,
                allowed_facts=services.brief_facts,
                on_report=lambda task, report, ok, _owner=owner: services.publish(
                    "verifier", _owner, MessageType.VERIFICATION_REPORT,
                    {"task": task, "passed": ok, **report.summary()},
                ),
            )
        guardrail = TaskGuardrail(
            name, verification=verification, max_attempts=settings.guardrail_max_retries + 1
        )

        tasks[name] = Task(
            name=name,
            description=description,
            expected_output=spec["expected_output"],
            # Pinning the owner restricts the manager's delegation tools to that
            # specialist: task-level least privilege on top of the RBAC hook.
            agent=workers[owner],
            context=[tasks[c] for c in context_names if c in tasks],
            guardrail=_as_function(guardrail),
            guardrail_max_retries=settings.guardrail_max_retries,
        )
        services.task_owner_roles[name] = workers[owner].role

    if not tasks:
        return CrewPlan(None, inputs, [], skipped, owners, role_by_display)

    manager = Agent(
        role=manager_spec["role"],
        goal=manager_spec["goal"],
        backstory=manager_spec["backstory"],
        llm=llm_factory(manager_key, manager_spec.get("effort")),
        allow_delegation=True,
        tools=[],
        verbose=verbose,
    )

    def on_task_complete(output: Any) -> None:
        name = getattr(output, "name", None) or ""
        owner = owners.get(name, "unknown")
        raw = getattr(output, "raw", "") or ""
        services.checkpoints.save(name, raw, owner=owner)
        services.memory.agent(services.session_id, owner).add_ai_message(f"[{name}] {raw[:4000]}")
        services.publish(owner, "manager", MessageType.TASK_COMPLETED, {"task": name, "chars": len(raw)})

    crew = Crew(
        agents=list(workers.values()),
        tasks=list(tasks.values()),
        process=Process.hierarchical,
        manager_agent=manager,
        memory=False,  # persistent memory is our Redis layer, not CrewAI's embeddings store
        planning=False,
        verbose=verbose,
        task_callback=on_task_complete,
        tracing=False,
    )
    return CrewPlan(crew, inputs, list(tasks), skipped, owners, role_by_display)
