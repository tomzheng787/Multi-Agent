"""Tool-augmented local search agent.

A self-contained agent that answers local-market questions over live data
using native Claude function calling. Tools come from the dynamic
:class:`ApiRegistry`; the loop itself is self-correcting:

* failed or empty tool calls return ``is_error`` results carrying a hint, so
  the model repairs its arguments on the next turn (counted as corrections);
* the final answer is a strict ``submit_report`` tool call validated with
  Pydantic and checked for fabricated evidence ids - a rejected report goes
  back to the model with the reason;
* a model that stops without submitting is nudged, and an exhausted step
  budget degrades to a partial report built from the evidence gathered.
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from marketing_crew.llm.claude import ClaudeClient, strict_json_schema
from marketing_crew.memory.store import EvidenceStore
from marketing_crew.tools.api_registry import ApiInvoker, outcome_to_json

log = logging.getLogger(__name__)

SUBMIT_TOOL_NAME = "submit_report"

SYSTEM_PROMPT = """\
You are a local-market research agent working for a marketing team. You answer questions about \
the businesses, competitors and market context around a specific place, using live data tools.

How to work:
- Resolve the place to coordinates with geocode_location before searching nearby.
- Query find_nearby_businesses once per relevant category. Use the encyclopedia tools for \
neighbourhood or industry context when it helps. Independent calls may run in parallel.
- Every successful tool result includes an evidence_id. Each business or fact you report must \
cite the evidence_id it came from.
- If a tool returns ok=false or no data, read its hint, change the arguments accordingly \
(broader radius, another category, a simpler place name, an exact article title) and try again. \
Never repeat an identical failing call.
- Never invent businesses, figures or evidence ids. Record anything you could not establish \
under data_gaps.
- Finish by calling submit_report exactly once."""


class Competitor(BaseModel):
    name: str
    category: str
    address: str | None
    distance_m: int | None
    evidence_id: str


class Insight(BaseModel):
    statement: str
    evidence_ids: list[str]


class LocalSearchReport(BaseModel):
    location: str = Field(description="Resolved place name")
    latitude: float | None
    longitude: float | None
    competitors: list[Competitor]
    insights: list[Insight] = Field(description="Market observations, each citing evidence ids")
    data_gaps: list[str] = Field(description="What could not be established from the data")

    def cited_ids(self) -> set[str]:
        ids = {c.evidence_id for c in self.competitors}
        for insight in self.insights:
            ids.update(insight.evidence_ids)
        return {i.strip().upper() for i in ids}


SUBMIT_TOOL = {
    "name": SUBMIT_TOOL_NAME,
    "description": (
        "Submit the final local-market report. Call exactly once, after gathering evidence. "
        "Cite only evidence_ids returned by earlier tool results."
    ),
    "input_schema": strict_json_schema(LocalSearchReport),
    "strict": True,
}


@dataclass
class ToolCallRecord:
    step: int
    tool: str
    args: dict[str, Any]
    ok: bool
    error: str | None = None
    evidence_id: str | None = None
    latency_ms: float = 0.0


@dataclass
class SearchRun:
    question: str
    report: LocalSearchReport | None
    status: str  # "completed" | "partial"
    steps: int
    self_corrections: int
    trace: list[ToolCallRecord] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "status": self.status,
                "steps": self.steps,
                "self_corrections": self.self_corrections,
                "tool_calls": len(self.trace),
                "failed_tool_calls": sum(1 for r in self.trace if not r.ok),
                "report": self.report.model_dump() if self.report else None,
            },
            ensure_ascii=False,
        )

    def trace_dicts(self) -> list[dict[str, Any]]:
        return [asdict(r) for r in self.trace]


class LocalSearchAgent:
    def __init__(
        self,
        llm: ClaudeClient,
        invoker: ApiInvoker,
        evidence: EvidenceStore,
        *,
        max_steps: int = 12,
        max_nudges: int = 2,
        effort: str | None = None,
    ):
        self.llm = llm
        self.invoker = invoker
        self.evidence = evidence
        self.max_steps = max_steps
        self.max_nudges = max_nudges
        self.effort = effort
        self.tools = invoker.registry.tool_definitions() + [SUBMIT_TOOL]

    def run(self, question: str) -> SearchRun:
        messages: list[dict[str, Any]] = [
            {
                "role": "user",
                "content": f"Research question: {question}\n\nYou have at most {self.max_steps} turns.",
            }
        ]
        trace: list[ToolCallRecord] = []
        last_failure: dict[str, dict[str, Any]] = {}
        corrections = nudges = 0

        for step in range(1, self.max_steps + 1):
            response = self.llm.create(messages=messages, system=SYSTEM_PROMPT, tools=self.tools, effort=self.effort)
            # Append the full content (thinking blocks included) unchanged.
            messages.append({"role": "assistant", "content": response.content})
            tool_uses = [b for b in response.content if getattr(b, "type", None) == "tool_use"]

            if not tool_uses:
                nudges += 1
                if nudges > self.max_nudges:
                    break
                messages.append(
                    {
                        "role": "user",
                        "content": "You have not submitted yet. Call submit_report now with what you "
                        "have; list anything unverified under data_gaps.",
                    }
                )
                continue

            data_calls = [b for b in tool_uses if b.name != SUBMIT_TOOL_NAME]
            submits = [b for b in tool_uses if b.name == SUBMIT_TOOL_NAME]
            results: list[dict[str, Any]] = []

            if data_calls:
                with ThreadPoolExecutor(max_workers=min(4, len(data_calls))) as pool:
                    executed = list(pool.map(lambda b: self._call_tool(step, b), data_calls))
                for block, (result, record) in zip(data_calls, executed):
                    trace.append(record)
                    results.append(result)
                    if not record.ok:
                        last_failure[block.name] = record.args
                    elif block.name in last_failure and last_failure.pop(block.name) != record.args:
                        corrections += 1  # same tool, repaired arguments, now succeeding

            for block in submits:
                report, error = self._accept_report(block.input)
                if report is not None:
                    return SearchRun(question, report, "completed", step, corrections, trace)
                results.append(
                    {"type": "tool_result", "tool_use_id": block.id, "content": error, "is_error": True}
                )

            # All results of one turn go back in a single user message.
            messages.append({"role": "user", "content": results})

        log.warning("local search budget exhausted after %s steps; returning partial report", self.max_steps)
        return SearchRun(question, self._partial_report(question, trace), "partial", self.max_steps, corrections, trace)

    # -- internals ----------------------------------------------------------
    def _call_tool(self, step: int, block: Any) -> tuple[dict[str, Any], ToolCallRecord]:
        args = dict(block.input or {})
        outcome = self.invoker.invoke(block.name, args)
        evidence_id = None
        if outcome.ok and not outcome.empty:
            evidence = self.evidence.add(
                source=block.name,
                title=f"{block.name} {json.dumps(args, sort_keys=True)}",
                url=outcome.source_urls[-1] if outcome.source_urls else None,
                content=json.dumps(outcome.data, ensure_ascii=False),
            )
            evidence_id = evidence.id
        record = ToolCallRecord(step, block.name, args, outcome.ok, outcome.error, evidence_id, outcome.latency_ms)
        result = {
            "type": "tool_result",
            "tool_use_id": block.id,
            "content": outcome_to_json(outcome, evidence_id=evidence_id),
        }
        if not outcome.ok:
            result["is_error"] = True
        return result, record

    def _accept_report(self, payload: Any) -> tuple[LocalSearchReport | None, str]:
        try:
            report = LocalSearchReport.model_validate(payload)
        except ValidationError as exc:
            return None, f"submit_report rejected - schema errors: {exc}. Fix them and submit again."
        unknown = sorted(eid for eid in report.cited_ids() if not self.evidence.exists(eid))
        if unknown:
            return None, (
                f"submit_report rejected - these evidence_ids were never returned by a tool: {unknown}. "
                "Cite only ids from tool results, or move the claim to data_gaps, then submit again."
            )
        return report, ""

    def _partial_report(self, question: str, trace: list[ToolCallRecord]) -> LocalSearchReport:
        competitors: list[Competitor] = []
        for record in trace:
            if record.tool != "find_nearby_businesses" or not record.evidence_id:
                continue
            evidence = self.evidence.get(record.evidence_id)
            for place in json.loads(evidence.content) if evidence else []:
                competitors.append(
                    Competitor(
                        name=place["name"], category=place["category"], address=place.get("address"),
                        distance_m=place.get("distance_m"), evidence_id=record.evidence_id,
                    )
                )
        return LocalSearchReport(
            location=question, latitude=None, longitude=None, competitors=competitors, insights=[],
            data_gaps=["Search budget ran out before the agent submitted; competitors were compiled "
                       "directly from tool output without analysis."],
        )
