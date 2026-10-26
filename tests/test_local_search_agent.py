import httpx

from marketing_crew.llm.offline import OfflineClaudeClient, _response, _tool_use, offline_http_transport
from marketing_crew.settings import CONFIG_DIR
from marketing_crew.tools.api_registry import ApiInvoker, ApiRegistry, HttpFetcher
from marketing_crew.tools.local_search_agent import LocalSearchAgent


def make_agent(settings, evidence, llm=None, **kwargs):
    fetcher = HttpFetcher(httpx.Client(transport=offline_http_transport()), user_agent="test")
    registry = ApiRegistry.from_yaml(CONFIG_DIR / "apis.yaml")
    for spec in registry.specs.values():
        spec.min_interval_s = 0
    return LocalSearchAgent(llm or OfflineClaudeClient(settings), ApiInvoker(registry, fetcher), evidence, **kwargs)


def test_self_corrects_and_submits_grounded_report(settings, evidence):
    run = make_agent(settings, evidence).run("Which cafes compete with cold brew in Mission District, San Francisco?")
    assert run.status == "completed"
    assert run.self_corrections == 1  # out-of-range radius repaired from the hint
    failed = [r for r in run.trace if not r.ok]
    assert len(failed) == 1 and failed[0].args["radius_m"] == 9000
    assert run.report.competitors[0].name == "Fogline Coffee Bar"
    assert all(evidence.exists(eid) for eid in run.report.cited_ids())


class ScriptedLLM:
    """Replays a fixed list of responses and records what it was sent."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def create(self, **kwargs):
        self.requests.append({**kwargs, "messages": list(kwargs["messages"])})  # agent mutates the list
        return self.responses.pop(0)


def test_rejects_fabricated_evidence_ids_then_accepts_fix(settings, evidence):
    report = {"location": "X", "latitude": None, "longitude": None, "competitors": [],
              "insights": [{"statement": "made up", "evidence_ids": ["E42"]}], "data_gaps": []}
    fixed = {**report, "insights": [], "data_gaps": ["nothing verifiable"]}
    llm = ScriptedLLM([_response([_tool_use("submit_report", report)], "tool_use"),
                       _response([_tool_use("submit_report", fixed)], "tool_use")])
    run = make_agent(settings, evidence, llm=llm).run("q")
    assert run.status == "completed" and run.steps == 2
    rejection = llm.requests[1]["messages"][-1]["content"][0]
    assert rejection["is_error"] and "E42" in rejection["content"]


def test_nudges_then_degrades_to_partial_report(settings, evidence):
    text = _response([type("B", (), {"type": "text", "text": "done"})()], "end_turn")
    llm = ScriptedLLM([text] * 3)
    run = make_agent(settings, evidence, llm=llm, max_nudges=2).run("q")
    assert run.status == "partial" and run.report.data_gaps


def test_step_budget_is_enforced(settings, evidence):
    call = _response([_tool_use("geocode_location", {"query": "Mission District", "limit": 1})], "tool_use")
    llm = ScriptedLLM([call] * 3)
    run = make_agent(settings, evidence, llm=llm, max_steps=3).run("q")
    assert run.status == "partial" and len(llm.requests) == 3
