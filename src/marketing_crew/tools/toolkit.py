"""Agent tools.

Tools are written once as LangChain ``StructuredTool``s (typed args schemas,
reusable outside CrewAI) and bridged into CrewAI by :class:`GuardedTool`,
which re-checks RBAC on every call before delegating to the LangChain tool.
Each tool set is built for one role, so tools know who is calling them.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from crewai.tools import BaseTool
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from marketing_crew.messaging.bus import AgentMessage, MessageType
from marketing_crew.security.rbac import AccessController, PermissionDenied

if TYPE_CHECKING:
    from marketing_crew.runtime import Services

log = logging.getLogger(__name__)

ALLOWED_ARTIFACT_SUFFIXES = {".md", ".json", ".csv", ".txt"}


# -- argument schemas ---------------------------------------------------------


class LocalMarketSearchInput(BaseModel):
    question: str = Field(description="Local-market question naming the place, e.g. 'Which cafes compete with a cold-brew bar in Mission District, San Francisco?'")


class WikipediaLookupInput(BaseModel):
    topic: str = Field(description="Topic to look up, e.g. a neighbourhood, industry or product category")


class EvidenceLookupInput(BaseModel):
    query: str = Field(default="", description="Keywords to search recorded evidence for")
    ids: list[str] = Field(default_factory=list, description="Specific evidence ids, e.g. ['E2', 'E5']")


class ChannelWeight(BaseModel):
    name: str
    weight: float = Field(gt=0, description="Relative priority; weights are normalised")
    min_amount: float = Field(default=0, ge=0, description="Floor in USD for this channel")


class BudgetAllocatorInput(BaseModel):
    total_budget: float = Field(gt=0)
    channels: list[ChannelWeight] = Field(min_length=1)


class VerifyClaimsInput(BaseModel):
    text: str = Field(description="The full text to fact-check against recorded evidence")


class SaveArtifactInput(BaseModel):
    filename: str = Field(description="File name only, e.g. 'launch_plan.md' (.md/.json/.csv/.txt)")
    content: str


class ScheduledItem(BaseModel):
    date: str = Field(description="ISO date YYYY-MM-DD")
    title: str
    format: str = Field(description="e.g. reel, email, event, blog post")
    call_to_action: str = ""


class ScheduleCampaignInput(BaseModel):
    channel: str
    items: list[ScheduledItem] = Field(min_length=1)


class SendAgentMessageInput(BaseModel):
    recipient: str = Field(description="Role key: researcher, analyst, writer, critic or executor")
    message_type: Literal["info", "feedback", "revision_request", "approval"] = "info"
    content: str


class ReadAgentInboxInput(BaseModel):
    max_messages: int = Field(default=10, ge=1, le=50)


class RecallMemoryInput(BaseModel):
    query: str = Field(default="", description="Optional focus for the recall")


# -- pure helpers ---------------------------------------------------------------


def allocate_budget(total_budget: float, channels: list[ChannelWeight]) -> list[dict[str, Any]]:
    """Split a budget by weight after honouring per-channel floors; sums exactly."""
    floors = sum(c.min_amount for c in channels)
    if floors > total_budget:
        raise ValueError(f"channel minimums ({floors:,.0f}) exceed the total budget ({total_budget:,.0f})")
    remaining = total_budget - floors
    weight_sum = sum(c.weight for c in channels)
    rows = []
    for c in channels:
        amount = round(c.min_amount + remaining * c.weight / weight_sum)
        rows.append({"channel": c.name, "amount_usd": amount})
    drift = round(total_budget) - sum(r["amount_usd"] for r in rows)
    rows[max(range(len(rows)), key=lambda i: rows[i]["amount_usd"])]["amount_usd"] += drift
    for row in rows:
        row["share_pct"] = round(100 * row["amount_usd"] / total_budget, 1)
    return rows


def safe_artifact_path(output_dir: Path, filename: str) -> Path:
    name = Path(filename).name  # drop any directory components
    if not re.fullmatch(r"[\w.\-]+", name) or Path(name).suffix.lower() not in ALLOWED_ARTIFACT_SUFFIXES:
        raise ValueError(
            f"invalid artifact name '{filename}'; use letters, digits, '-', '_' and one of "
            f"{sorted(ALLOWED_ARTIFACT_SUFFIXES)}"
        )
    return output_dir / name


# -- LangChain tools ------------------------------------------------------------


def build_langchain_tools(services: "Services", role: str) -> dict[str, StructuredTool]:
    """Every tool in the catalogue, bound to ``role``. RBAC decides who gets which."""

    def local_market_search(question: str) -> str:
        run = services.search_agent.run(question)
        return run.to_json()

    def wikipedia_lookup(topic: str) -> str:
        found = services.invoker.invoke("wikipedia_search", {"query": topic, "limit": 3})
        if not found.ok or not found.data:
            return json.dumps(found.to_payload())
        title = found.data[0]["title"]
        summary = services.invoker.invoke("wikipedia_summary", {"title": title})
        if not summary.ok or not summary.data:
            return json.dumps(summary.to_payload())
        evidence = services.evidence.add(
            source="wikipedia_summary", title=title, url=(summary.data or {}).get("url"),
            content=summary.data.get("extract") or "",
        )
        return json.dumps({"evidence_id": evidence.id, "title": title, "extract": summary.data.get("extract")})

    def evidence_lookup(query: str = "", ids: list[str] | None = None) -> str:
        items = services.evidence.get_many(ids) if ids else services.evidence.search(query, k=6)
        return services.evidence.render(items, max_chars=800)

    def budget_allocator(total_budget: float, channels: list[ChannelWeight | dict]) -> str:
        parsed = [c if isinstance(c, ChannelWeight) else ChannelWeight.model_validate(c) for c in channels]
        rows = allocate_budget(total_budget, parsed)
        table = "\n".join(f"| {r['channel']} | ${r['amount_usd']:,} | {r['share_pct']}% |" for r in rows)
        text = f"| Channel | Budget | Share |\n|---|---|---|\n{table}"
        evidence = services.evidence.add(source="budget_allocator", title=f"Budget split of ${total_budget:,.0f}", content=text)
        return f"{text}\n\nRecorded as evidence {evidence.id}."

    def verify_claims(text: str) -> str:
        report = services.verifier.verify(text, allowed_facts=services.brief_facts)
        services.publish(
            role, "*", MessageType.VERIFICATION_REPORT, {"summary": report.summary()}
        )
        if not report.flagged:
            return f"All {len(report.claims)} claims verified. {json.dumps(report.summary())}"
        return report.feedback(services.settings.max_hallucination_rate)

    def save_artifact(filename: str, content: str) -> str:
        path = safe_artifact_path(services.output_dir, filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
        services.artifacts.add(str(path))
        return f"Saved {len(content):,} characters to {path}"

    def schedule_campaign(channel: str, items: list[ScheduledItem | dict]) -> str:
        parsed = [i if isinstance(i, ScheduledItem) else ScheduledItem.model_validate(i) for i in items]
        for item in parsed:
            date.fromisoformat(item.date)  # raises on malformed dates -> model retries
        path = services.output_dir / "campaign_schedule.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        schedule = json.loads(path.read_text()) if path.exists() else {}
        schedule.setdefault(channel, []).extend(i.model_dump() for i in parsed)
        path.write_text(json.dumps(schedule, indent=2))
        services.artifacts.add(str(path))
        # A real integration (Buffer, HubSpot, Mailchimp...) plugs in here.
        return f"Scheduled {len(parsed)} item(s) on '{channel}' (local schedule file {path}; nothing was published)."

    def send_agent_message(recipient: str, content: str, message_type: str = "info") -> str:
        recipient = recipient.strip().lower()
        if recipient not in services.access.policy.roles or recipient in {"base", "manager", role}:
            return f"Unknown recipient '{recipient}'. Use one of: researcher, analyst, writer, critic, executor."
        message_id = services.publish(role, recipient, MessageType(message_type), {"content": content})
        return f"Message {message_id} queued for {recipient}."

    def read_agent_inbox(max_messages: int = 10) -> str:
        deliveries = services.bus.consume(role, f"{role}-inbox", role, count=max_messages)
        lines = []
        for delivery in deliveries:
            message = delivery.message
            lines.append(f"- from {message.sender} ({message.type.value}): {message.payload.get('content', message.payload)}")
            services.bus.ack(role, f"{role}-inbox", delivery.entry_id)
        return "\n".join(lines) or "Inbox empty."

    def recall_memory(query: str = "") -> str:
        return services.memory.recall(services.session_id, role)

    specs: list[tuple[Any, str, type[BaseModel], str]] = [
        (local_market_search, "local_market_search", LocalMarketSearchInput,
         "Run the autonomous local search agent: geocodes the place, maps nearby competitors from live "
         "OpenStreetMap data and pulls encyclopedia context, self-correcting failed calls. Returns a JSON "
         "report whose facts carry evidence ids. Use it for any question about a specific location's market."),
        (wikipedia_lookup, "wikipedia_lookup", WikipediaLookupInput,
         "Look up background on a neighbourhood, industry or product category on Wikipedia and record it "
         "as evidence. Returns the evidence id and extract."),
        (evidence_lookup, "evidence_lookup", EvidenceLookupInput,
         "Search or fetch recorded evidence by keywords or ids. Use it before stating a fact, to find the "
         "evidence id to cite."),
        (budget_allocator, "budget_allocator", BudgetAllocatorInput,
         "Split a total budget across channels by relative weight with optional per-channel minimums. "
         "Exact arithmetic; the result is recorded as evidence. Use it instead of computing splits yourself."),
        (verify_claims, "verify_claims", VerifyClaimsInput,
         "Fact-check a text against recorded evidence with the chain-of-verification protocol. Returns "
         "unsupported or contradicted claims with reasons."),
        (save_artifact, "save_artifact", SaveArtifactInput,
         "Save a deliverable file (markdown/JSON/CSV/text) to this run's output folder."),
        (schedule_campaign, "schedule_campaign", ScheduleCampaignInput,
         "Add dated content items for one channel to the campaign schedule (local file; nothing is published)."),
        (send_agent_message, "send_agent_message", SendAgentMessageInput,
         "Send a structured message to another specialist's inbox (researcher, analyst, writer, critic, executor)."),
        (read_agent_inbox, "read_agent_inbox", ReadAgentInboxInput,
         "Read and acknowledge messages other agents sent you. Check it at the start of your task."),
        (recall_memory, "recall_memory", RecallMemoryInput,
         "Recall persistent memory from earlier runs in this session: previous briefs, deliverables and your notes."),
    ]
    return {
        name: StructuredTool.from_function(func=func, name=name, description=description, args_schema=schema)
        for func, name, schema, description in specs
    }


# -- CrewAI bridge --------------------------------------------------------------


class GuardedTool(BaseTool):
    """CrewAI tool that enforces RBAC, then runs the wrapped LangChain tool."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    lc_tool: Any = Field(exclude=True)
    access: Any = Field(exclude=True)
    role: str

    def _run(self, **kwargs: Any) -> str:
        try:
            self.access.check_tool(self.role, self.lc_tool.name)
        except PermissionDenied as exc:
            return f"ACCESS DENIED: {exc}. This action is outside your role; leave it to the responsible specialist."
        try:
            result = self.lc_tool.invoke(kwargs)
        except ValidationError as exc:
            return f"Invalid input for {self.name}: {exc}. Fix the Action Input and try again."
        except Exception as exc:
            log.warning("tool %s failed for %s: %s", self.name, self.role, exc)
            return f"Tool error ({type(exc).__name__}): {exc}. Adjust the input and retry, or continue without it."
        return result if isinstance(result, str) else json.dumps(result, ensure_ascii=False, default=str)


def to_crewai_tool(lc_tool: StructuredTool, *, role: str, access: AccessController) -> GuardedTool:
    return GuardedTool(
        name=lc_tool.name, description=lc_tool.description, args_schema=lc_tool.args_schema,
        lc_tool=lc_tool, access=access, role=role,
    )


def build_agent_tools(services: "Services", role: str, requested: list[str]) -> list[GuardedTool]:
    catalogue = build_langchain_tools(services, role)
    unknown = sorted(set(requested) - set(catalogue))
    if unknown:
        raise ValueError(f"agent '{role}' requests unknown tools: {unknown}")
    granted = services.access.tools_for(role, requested)
    denied = sorted(set(requested) - set(granted))
    if denied:
        log.warning("RBAC withheld tools %s from role '%s'", denied, role)
    return [to_crewai_tool(catalogue[name], role=role, access=services.access) for name in granted]
