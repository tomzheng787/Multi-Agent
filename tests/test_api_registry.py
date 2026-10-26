import httpx
import pytest

from marketing_crew.settings import CONFIG_DIR
from marketing_crew.tools.api_registry import ApiInvoker, ApiRegistry, ApiSpec, CircuitBreaker, HttpFetcher


def make_invoker(handler, specs=None, **kwargs):
    registry = ApiRegistry(specs) if specs else ApiRegistry.from_yaml(CONFIG_DIR / "apis.yaml")
    fetcher = HttpFetcher(httpx.Client(transport=httpx.MockTransport(handler)), user_agent="test",
                          base_delay_s=0.01, max_delay_s=0.02)
    return ApiInvoker(registry, fetcher, **kwargs)


ECHO = ApiSpec(
    name="echo", description="echo", url="https://api.test/items/{item}",
    query={"q": "{query}", "fixed": "1"},
    params_schema={"type": "object", "properties": {"item": {"type": "string"}, "query": {"type": "string"}},
                   "required": ["item"], "additionalProperties": False},
    result_path="data.items", result_fields={"name": "name", "city": "address.city"}, max_items=2,
)


def test_registry_exposes_strict_tool_definitions():
    registry = ApiRegistry.from_yaml(CONFIG_DIR / "apis.yaml")
    tools = {t["name"]: t for t in registry.tool_definitions()}
    assert {"geocode_location", "find_nearby_businesses", "wikipedia_search", "wikipedia_summary"} <= set(tools)
    radius = tools["find_nearby_businesses"]["input_schema"]["properties"]["radius_m"]
    assert "maximum" not in radius  # unsupported by strict tools; enforced client-side instead
    assert all(t["strict"] and t["input_schema"]["additionalProperties"] is False for t in tools.values())


def test_templating_and_extraction():
    seen = {}

    def handler(request):
        seen["url"] = str(request.url)
        return httpx.Response(200, json={"data": {"items": [
            {"name": "A", "address": {"city": "SF"}}, {"name": "B", "address": {"city": "LA"}}, {"name": "C"}]}})

    outcome = make_invoker(handler, [ECHO]).invoke("echo", {"item": "a b/c"})
    assert outcome.ok and outcome.data == [{"name": "A", "city": "SF"}, {"name": "B", "city": "LA"}]
    assert seen["url"] == "https://api.test/items/a%20b%2Fc?fixed=1"  # optional {query} omitted, path quoted


def test_invalid_arguments_return_hint_without_network_call():
    calls = []
    invoker = make_invoker(lambda r: calls.append(r) or httpx.Response(200, json=[]))
    outcome = invoker.invoke("find_nearby_businesses", {"lat": 1.0, "lon": 2.0, "category": "cafe", "radius_m": 9000})
    assert not outcome.ok and "radius_m" in outcome.error and outcome.hint
    assert calls == []


def test_retries_transient_errors_then_succeeds():
    responses = iter([httpx.Response(503), httpx.Response(429, headers={"retry-after": "0"}),
                      httpx.Response(200, json=[{"display_name": "X", "lat": "1", "lon": "2"}])])
    outcome = make_invoker(lambda r: next(responses)).invoke("geocode_location", {"query": "X"})
    assert outcome.ok and outcome.data[0]["display_name"] == "X"


def test_empty_results_carry_correction_hint():
    outcome = make_invoker(lambda r: httpx.Response(200, json=[])).invoke("geocode_location", {"query": "nowhere"})
    assert outcome.ok and outcome.empty and "broader" in outcome.hint


def test_unknown_category_hint_from_python_tool():
    from marketing_crew.tools.api_registry import ToolInputError
    from marketing_crew.tools.osm import find_places

    with pytest.raises(ToolInputError) as err:
        find_places(lambda *a, **k: {}, lat=1, lon=2, category="coffee shop")
    assert "cafe" in err.value.hint


def test_circuit_breaker_opens_and_half_opens():
    now = [0.0]
    breaker = CircuitBreaker(threshold=2, reset_s=10, clock=lambda: now[0])
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == "open"
    now[0] = 11
    assert breaker.state == "half_open"
    breaker.record_success()
    assert breaker.state == "closed"


def test_invoker_fails_fast_when_circuit_open():
    calls = []
    invoker = make_invoker(lambda r: calls.append(1) or httpx.Response(503), breaker_threshold=1)
    first = invoker.invoke("geocode_location", {"query": "X"})
    second = invoker.invoke("geocode_location", {"query": "X"})
    assert not first.ok and "upstream unavailable" in first.error
    assert not second.ok and "circuit open" in second.error
    assert len(calls) == 3  # retries on the first call only
