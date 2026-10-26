"""Offline mode: deterministic stand-ins for Claude and the live data APIs.

Runs the whole system - hierarchical delegation, RBAC, ReAct tool calls, the
self-correcting search loop, verification retries, checkpoints and the
message bus - with no network access and no API key. Agent outputs are
scripted, not generated, and the HTTP fixtures (businesses, coordinates,
encyclopedia text) are synthetic. Use it for demos, CI and development.

The script deliberately exercises the fault-handling paths: the search agent
first sends an out-of-range radius and repairs it from the error hint, and the
writer's first draft contains a fabricated statistic that the verification
guardrail rejects.
"""

from __future__ import annotations

import json
import re
import uuid
from datetime import date, timedelta
from types import SimpleNamespace
from typing import Any
from urllib.parse import parse_qs, unquote

import httpx
from crewai import BaseLLM
from pydantic import ConfigDict, PrivateAttr

from marketing_crew.llm.claude import ClaudeClient, StructuredOutputError, UsageTracker
from marketing_crew.settings import Settings
from marketing_crew.tools.toolkit import ChannelWeight, allocate_budget
from marketing_crew.verification.cove import FEEDBACK_MARKER

# -- synthetic HTTP fixtures --------------------------------------------------------

_CENTER = (37.7599, -122.4148)
_FIXTURE_CAFES = [
    ("Fogline Coffee Bar", 37.7605, -122.4140, "Valencia Street"),
    ("Mission Bean Collective", 37.7570, -122.4180, "18th Street"),
    ("Valencia Street Roasters", 37.7590, -122.4210, "Valencia Street"),
    ("Twenty-Fourth Espresso", 37.7524, -122.4165, "24th Street"),
    ("Dolores Cold Brew Co.", 37.7615, -122.4260, "Dolores Street"),
]
_FIXTURE_EXTRACT = (
    "Synthetic offline fixture. Cold brew coffee is made by steeping ground coffee in cold water "
    "for 12 to 24 hours, which gives a smoother, less acidic drink than hot brewing. The surrounding "
    "district is a dense, mixed-use neighbourhood known for independent cafes, street murals and many "
    "young professionals."
)


def offline_http_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        host, path = request.url.host, request.url.path
        if host == "nominatim.openstreetmap.org":
            query = request.url.params.get("q", "")
            return httpx.Response(200, json=[{
                "display_name": f"{query} (synthetic fixture)", "lat": str(_CENTER[0]), "lon": str(_CENTER[1]),
                "type": "suburb", "importance": 0.62,
            }])
        if host == "overpass-api.de":
            query = parse_qs(request.content.decode()).get("data", [""])[0]
            elements = []
            if '"amenity"="cafe"' in query:
                elements = [
                    {"type": "node", "id": 1000 + i, "lat": lat, "lon": lon,
                     "tags": {"name": name, "amenity": "cafe", "addr:street": street}}
                    for i, (name, lat, lon, street) in enumerate(_FIXTURE_CAFES)
                ]
            return httpx.Response(200, json={"elements": elements})
        if host == "en.wikipedia.org" and path == "/w/api.php":
            return httpx.Response(200, json={"query": {"search": [
                {"title": "Cold brew coffee", "snippet": "<span>Cold brew</span> coffee is steeped in cold water"},
            ]}})
        if host == "en.wikipedia.org" and path.startswith("/api/rest_v1/page/summary/"):
            title = unquote(path.rsplit("/", 1)[-1])
            return httpx.Response(200, json={
                "title": title, "description": "Synthetic offline fixture", "extract": _FIXTURE_EXTRACT,
                "content_urls": {"desktop": {"page": f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}"}},
            })
        return httpx.Response(404, text="no offline fixture for this request")

    return httpx.MockTransport(handler)


# -- scripted Claude for the local search agent ----------------------------------------


def _block(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(**fields)


def _tool_use(name: str, payload: dict[str, Any]) -> SimpleNamespace:
    return _block(type="tool_use", id=f"toolu_{uuid.uuid4().hex[:12]}", name=name, input=payload)


def _response(content: list[Any], stop_reason: str) -> SimpleNamespace:
    return SimpleNamespace(content=content, stop_reason=stop_reason, usage=None, stop_details=None)


class OfflineClaudeClient(ClaudeClient):
    """Scripted policy standing in for Claude inside the local search agent."""

    def __init__(self, settings: Settings):  # no SDK client, no credentials needed
        self.settings = settings
        self._client = None
        self.usage = UsageTracker()

    def create(self, *, messages: list[dict[str, Any]], tools: list[dict[str, Any]] | None = None, **_: Any) -> Any:
        if not any(t["name"] == "submit_report" for t in tools or []):
            return _response([_block(type="text", text="(offline mode)")], "end_turn")
        return self._search_policy(messages)

    def structured(self, *_: Any, **__: Any) -> Any:
        raise StructuredOutputError("offline mode has no structured generation")

    @staticmethod
    def _results(messages: list[dict[str, Any]]) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
        calls = {
            b.id: (b.name, b.input)
            for m in messages if m["role"] == "assistant"
            for b in m["content"] if getattr(b, "type", None) == "tool_use"
        }
        out = []
        for m in messages:
            if m["role"] == "user" and isinstance(m["content"], list):
                for r in m["content"]:
                    name, args = calls[r["tool_use_id"]]
                    out.append((name, args, json.loads(r["content"])))
        return out

    def _search_policy(self, messages: list[dict[str, Any]]) -> Any:
        question = messages[0]["content"].split("\n")[0].removeprefix("Research question: ")
        match = re.search(r"\b(?:around|in|near)\s+(.+?)[?.]?$", question)
        place = match.group(1) if match else question
        turn = sum(1 for m in messages if m["role"] == "assistant")
        results = self._results(messages)

        if turn == 0:
            return _response([_tool_use("geocode_location", {"query": place, "limit": 3})], "tool_use")
        geo = next(r for n, _, r in results if n == "geocode_location" and r["ok"])
        lat, lon = float(geo["data"][0]["lat"]), float(geo["data"][0]["lon"])
        if turn == 1:
            # Deliberate mistake: radius outside the allowed range -> error + hint.
            return _response([_tool_use("find_nearby_businesses", {
                "lat": lat, "lon": lon, "category": "cafe", "radius_m": 9000, "limit": 10})], "tool_use")
        if turn == 2:
            return _response([
                _tool_use("find_nearby_businesses", {"lat": lat, "lon": lon, "category": "cafe", "radius_m": 1500, "limit": 10}),
                _tool_use("wikipedia_summary", {"title": "Cold brew coffee"}),
            ], "tool_use")

        places = next(r for n, _, r in results if n == "find_nearby_businesses" and r["ok"])
        wiki = next(r for n, _, r in results if n == "wikipedia_summary" and r["ok"])
        nearest = places["data"][0]
        report = {
            "location": geo["data"][0]["display_name"],
            "latitude": lat,
            "longitude": lon,
            "competitors": [
                {"name": p["name"], "category": p["category"], "address": p["address"],
                 "distance_m": p["distance_m"], "evidence_id": places["evidence_id"]}
                for p in places["data"]
            ],
            "insights": [
                {"statement": f"The nearest cafe, {nearest['name']}, is {nearest['distance_m']} m from the search centre.",
                 "evidence_ids": [places["evidence_id"]]},
                {"statement": "Cold brew is steeped for 12 to 24 hours and tastes smoother and less acidic.",
                 "evidence_ids": [wiki["evidence_id"]]},
            ],
            "data_gaps": ["Open sources give no foot-traffic, pricing or sales figures."],
        }
        return _response([_tool_use("submit_report", report)], "tool_use")


# -- scripted CrewAI agents ------------------------------------------------------------

CHANNELS = [("Instagram & TikTok", 40), ("Pop-up tastings", 25), ("Office partnerships", 20), ("Email & SMS", 15)]


class ScriptedCrewLLM(BaseLLM):
    """Deterministic ReAct agent used for every crew role in offline mode."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_type: str = "scripted"
    rbac_role: str
    _services: Any = PrivateAttr()

    def __init__(self, *, services: Any, rbac_role: str, **kwargs: Any):
        super().__init__(model="offline-scripted", provider="offline", rbac_role=rbac_role, **kwargs)
        self._services = services

    def supports_function_calling(self) -> bool:
        return False

    def call(self, messages, tools=None, callbacks=None, available_functions=None,
             from_task=None, from_agent=None, response_model=None) -> str:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        prompt = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "user")
        actions = "\n".join(str(m.get("content", "")) for m in messages if m.get("role") == "assistant")
        # Count our own action marker: CrewAI periodically re-injects its tool
        # instructions (which contain "Observation:") into observations.
        step = actions.count(_ACTION_MARKER)
        handler = getattr(self, f"_{self.rbac_role}", None)
        if handler is None:
            return _final("(offline) acknowledged.")
        return handler(step=step, prompt=prompt, actions=actions, task=from_task)

    # -- evidence helpers -------------------------------------------------------
    @property
    def brief(self) -> dict[str, str]:
        return self._services.brief

    def _evidence(self, source: str):
        return next((e for e in reversed(self._services.evidence.all()) if e.source == source), None)

    def _places(self) -> tuple[str, list[dict[str, Any]]]:
        ev = self._evidence("find_nearby_businesses")
        return (ev.id, json.loads(ev.content)) if ev else ("", [])

    # -- roles ------------------------------------------------------------------
    def _manager(self, *, step: int, prompt: str, actions: str, task: Any) -> str:
        if step and "Delegate work to coworker" in actions:
            last = actions.rsplit(_ACTION_MARKER, 1)[1]
            observation = last.split("Observation:", 1)[1] if "Observation:" in last else ""
            # Drop CrewAI's periodic tool-instructions reminder if it was appended.
            return _final(observation.split("\nYou ONLY have access", 1)[0].strip())
        name = getattr(task, "name", None) or ""
        coworker = self._services.task_owner_roles.get(name, "Market Researcher")
        title = name.replace("_", " ") or "task"
        return _action("Delegate work to coworker", {"task": f"Complete the {title} task.", "context": prompt, "coworker": coworker})

    def _researcher(self, *, step: int, **_: Any) -> str:
        b = self.brief
        if step == 0:
            return _action("local_market_search", {"question": f"Which cafes compete with {b['product']} around {b['location']}?"})
        if step == 1:
            return _action("send_agent_message", {
                "recipient": "analyst", "message_type": "info",
                "content": "Data gap: open sources have no foot-traffic or pricing data; treat demand as an assumption.",
            })
        geo, wiki = self._evidence("geocode_location"), self._evidence("wikipedia_summary")
        places_id, places = self._places()
        rows = "\n".join(f"| {p['name']} | {p['category']} | {p['distance_m']} m | [{places_id}] |" for p in places)
        return _final(
            f"## Location profile\n{b['location']} resolves to {json.loads(geo.content)[0]['display_name']} [{geo.id}].\n"
            f"Cold brew is steeped in cold water for 12 to 24 hours, giving a smoother, less acidic drink [{wiki.id}].\n\n"
            f"## Competitors\n| Name | Category | Distance | Evidence |\n|---|---|---|---|\n{rows}\n\n"
            "## Data gaps\n- Open sources give no foot-traffic, pricing or sales data."
        )

    def _analyst(self, *, step: int, **_: Any) -> str:
        if step == 0:
            return _action("read_agent_inbox", {"max_messages": 10})
        b = self.brief
        places_id, places = self._places()
        first, second = places[0], places[1]
        wiki = self._evidence("wikipedia_summary")
        return _final(
            "## Competitive landscape\n"
            f"The closest competitor, {first['name']}, is {first['distance_m']} m from the search centre [{places_id}].\n"
            f"{second['name']} is {second['distance_m']} m away [{places_id}].\n\n"
            "## SWOT\n"
            f"- Strength: cold brew is smoother and less acidic than hot-brewed coffee [{wiki.id}].\n"
            f"- Weakness (analysis): {b['brand']} has no local awareness yet.\n"
            "- Opportunity (analysis): no mapped competitor presents itself as a cold-brew specialist.\n"
            f"- Threat: {first['name']} ({first['category']}, {first['distance_m']} m away) already serves nearby coffee drinkers [{places_id}].\n\n"
            f"## Target segments\n- {b['audience']}\n\n"
            f"## Recommended positioning\nWe recommend positioning {b['brand']} as the neighbourhood's cold-brew specialist."
        )

    def _strategy(self, *, include_fabrication: bool) -> str:
        b = self.brief
        places_id, places = self._places()
        wiki = self._evidence("wikipedia_summary")
        budget = float(b["budget_usd"].replace(",", ""))
        split = allocate_budget(budget, [ChannelWeight(name=n, weight=w) for n, w in CHANNELS])
        budget_lines = "\n".join(f"- Allocate ${r['amount_usd']:,} to {r['channel']}." for r in split)
        fabricated = (
            f"Local surveys show 87% of residents buy cold brew every week [E99].\n" if include_fabrication else ""
        )
        return (
            f"# {b['brand']} marketing strategy\n\n"
            "## Objectives\n- Target 1,500 cold-brew trial redemptions by the end of the period.\n"
            "- Target a 25% repeat-purchase rate among trial customers.\n\n"
            f"## Positioning\n{b['brand']} is the cold-brew specialist: coffee steeped for 12 to 24 hours for a "
            f"smoother, less acidic drink [{wiki.id}].\n\n"
            f"## Market context\nThe nearest competitor, {places[0]['name']}, is {places[0]['distance_m']} m from the "
            f"search centre [{places_id}].\n{fabricated}\n"
            "## Messaging pillars\n- Smooth by design.\n- Brewed slow, served fast.\n- Your neighbourhood cold-brew bar.\n\n"
            f"## Channel mix and budget\n{budget_lines}\n\n"
            "## Roadmap\n- First phase: launch with pop-up tastings and creator content.\n"
            "- Second phase: scale office partnerships and email offers.\n"
            "- Third phase: shift spend toward the best-converting channel.\n\n"
            "## KPIs\n- KPI: trial redemptions, repeat-purchase rate, cost per acquired customer."
        )

    def _writer(self, *, prompt: str, **_: Any) -> str:
        return _final(self._strategy(include_fabrication=FEEDBACK_MARKER not in prompt))

    def _critic(self, *, step: int, **_: Any) -> str:
        strategy = self._strategy(include_fabrication=False)
        if step == 0:
            return _action("verify_claims", {"text": strategy})
        if step == 1:
            return _action("send_agent_message", {
                "recipient": "executor", "message_type": "approval",
                "content": f"Approved. Keep total spend at ${self.brief['budget_usd']}; lead with the cold-brew "
                           "specialist positioning; publish no statistic without a source.",
            })
        return _final(
            f"{strategy}\n\n## Changelog\n- Removed an unsourced survey statistic flagged by verification.\n"
            "- Confirmed the channel allocations add up to the total budget."
        )

    def _executor(self, *, step: int, **_: Any) -> str:
        b = self.brief
        budget = float(b["budget_usd"].replace(",", ""))
        start = date.today() + timedelta(days=7)
        if step == 0:
            return _action("read_agent_inbox", {"max_messages": 10})
        if step == 1:
            return _action("budget_allocator", {
                "total_budget": budget, "channels": [{"name": n, "weight": w} for n, w in CHANNELS]})
        if step == 2:
            return _action("schedule_campaign", {"channel": "Instagram & TikTok", "items": [
                {"date": (start + timedelta(days=d)).isoformat(), "title": t, "format": f, "call_to_action": "Claim a free tasting"}
                for d, t, f in [(0, "Meet the slow brew", "reel"), (3, "12 to 24 hours, explained", "carousel"),
                                (7, "Pop-up tasting this weekend", "story")]
            ]})
        if step == 3:
            return _action("save_artifact", {"filename": "strategy.md", "content": self._strategy(include_fabrication=False)})
        if step == 4:
            return _action("save_artifact", {"filename": "launch_plan.md", "content": (
                f"# {b['brand']} launch plan\n\nFirst content goes live on {start.isoformat()}.\n"
                "See campaign_schedule.json for dated items and the budget table recorded as evidence."
            )})
        return _final(
            "Execution summary: allocated the budget across four channels with budget_allocator, scheduled three "
            "Instagram & TikTok items, and saved strategy.md and launch_plan.md to the run folder."
        )


_ACTION_MARKER = "Thought: Next step - using "


def _action(tool: str, payload: dict[str, Any]) -> str:
    return f"{_ACTION_MARKER}{tool}.\nAction: {tool}\nAction Input: {json.dumps(payload)}"


def _final(text: str) -> str:
    return f"Thought: I now can give a great answer\nFinal Answer: {text}"
