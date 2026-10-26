from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from marketing_crew.llm.claude import (
    FALLBACK_BETA,
    ClaudeClient,
    LLMRefusalError,
    StructuredOutputError,
    strict_json_schema,
)
from marketing_crew.llm.crewai_adapter import ClaudeCrewLLM, to_anthropic_messages
from marketing_crew.settings import Settings

USAGE = SimpleNamespace(input_tokens=10, output_tokens=5, cache_read_input_tokens=3, cache_creation_input_tokens=1)


def response(text="hi", stop_reason="end_turn", stop_details=None, parsed=None):
    content = [SimpleNamespace(type="thinking", thinking=""), SimpleNamespace(type="text", text=text)]
    return SimpleNamespace(content=content, stop_reason=stop_reason, stop_details=stop_details,
                           usage=USAGE, parsed_output=parsed)


class FakeAnthropic:
    def __init__(self, reply):
        self.calls = []
        api = SimpleNamespace(create=self._record(reply), parse=self._record(reply))
        self.messages = api
        self.beta = SimpleNamespace(messages=api)

    def _record(self, reply):
        def call(**kwargs):
            self.calls.append(kwargs)
            return reply
        return call


def test_request_shape_with_fallbacks_effort_and_caching():
    fake = FakeAnthropic(response("hello"))
    client = ClaudeClient(Settings(effort="medium"), client=fake)
    assert client.complete("hi", system="sys") == "hello"
    kwargs = fake.calls[0]
    assert kwargs["model"] == "claude-opus-5-5"
    assert kwargs["output_config"] == {"effort": "medium"}
    assert kwargs["betas"] == [FALLBACK_BETA] and kwargs["fallbacks"] == "default"
    assert kwargs["cache_control"] == {"type": "ephemeral"}
    assert not {"temperature", "top_p", "top_k", "thinking"} & set(kwargs)
    assert client.usage.as_dict()["cache_read_tokens"] == 3


def test_fallbacks_can_be_disabled():
    fake = FakeAnthropic(response())
    ClaudeClient(Settings(use_fallbacks=False), client=fake).complete("hi")
    assert "betas" not in fake.calls[0] and "fallbacks" not in fake.calls[0]


def test_refusal_raises_with_category():
    fake = FakeAnthropic(response(stop_reason="refusal", stop_details=SimpleNamespace(category="cyber", explanation="x")))
    with pytest.raises(LLMRefusalError) as err:
        ClaudeClient(Settings(), client=fake).complete("hi")
    assert err.value.category == "cyber"


class Item(BaseModel):
    name: str


def test_structured_outputs():
    fake = FakeAnthropic(response(parsed=Item(name="x")))
    assert ClaudeClient(Settings(), client=fake).structured(Item, prompt="p").name == "x"
    assert fake.calls[0]["output_format"] is Item
    with pytest.raises(StructuredOutputError):
        ClaudeClient(Settings(), client=FakeAnthropic(response(parsed=None))).structured(Item, prompt="p")


def test_strict_schema_rewrites_constraints():
    schema = strict_json_schema({"type": "object", "properties": {
        "n": {"type": "integer", "minimum": 1, "title": "N"},
        "child": {"type": "object", "properties": {"title": {"type": "string"}}}}})
    assert schema["additionalProperties"] is False and schema["properties"]["child"]["additionalProperties"] is False
    assert schema["properties"]["n"] == {"type": "integer"}
    assert "title" in schema["properties"]["child"]["properties"]  # a property named "title" survives


def test_message_conversion():
    system, turns = to_anthropic_messages([
        {"role": "system", "content": "You are X."},
        {"role": "assistant", "content": "earlier"},
        {"role": "user", "content": "a"},
        {"role": "user", "content": "b"},
        {"role": "assistant", "content": "Thought: ...\nObservation: ok"},
    ])
    assert system == "You are X."
    assert [t["role"] for t in turns] == ["user", "assistant", "user", "assistant", "user"]
    assert turns[2]["content"] == "a\n\nb"


def test_crewai_llm_uses_stop_sequences_and_text_protocol():
    fake = FakeAnthropic(response("Thought: go\nAction: x\nAction Input: {}\nObservation: leaked"))
    llm = ClaudeCrewLLM(ClaudeClient(Settings(), client=fake), effort="high", stop=["\nObservation:"])
    out = llm.call([{"role": "system", "content": "sys"}, {"role": "user", "content": "task"}])
    assert out.endswith("Action Input: {}")  # truncated at the stop word
    kwargs = fake.calls[0]
    assert kwargs["stop_sequences"] == ["\nObservation:"]
    assert kwargs["output_config"]["effort"] == "high"
    assert kwargs["system"].startswith("sys")
    assert llm.supports_function_calling() is False
