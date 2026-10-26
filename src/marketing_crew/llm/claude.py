"""Thin wrapper around the Anthropic SDK shared by every agent.

Centralises the request shape (model, effort, refusal fallbacks, prompt
caching), refusal handling, structured outputs and token accounting so the
CrewAI adapter, the local search agent and the verifier all behave the same.
"""

from __future__ import annotations

import copy
import logging
import threading
from dataclasses import dataclass, field
from typing import Any, TypeVar

import anthropic
from pydantic import BaseModel

from marketing_crew.settings import Settings

log = logging.getLogger(__name__)

FALLBACK_BETA = "server-side-fallback-2026-07-01"

T = TypeVar("T", bound=BaseModel)

# JSON-schema keywords structured outputs / strict tools do not accept. They are
# stripped from what we send and enforced client-side instead.
_UNSUPPORTED_SCHEMA_KEYS = {
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
    "minLength", "maxLength", "minItems", "maxItems", "pattern", "default", "title",
}


class LLMRefusalError(RuntimeError):
    def __init__(self, category: str | None, explanation: str | None):
        self.category = category
        self.explanation = explanation
        super().__init__(f"model declined the request (category={category}): {explanation}")


class StructuredOutputError(RuntimeError):
    pass


@dataclass
class UsageTracker:
    calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, usage: Any) -> None:
        if usage is None:
            return
        with self._lock:
            self.calls += 1
            self.input_tokens += getattr(usage, "input_tokens", 0) or 0
            self.output_tokens += getattr(usage, "output_tokens", 0) or 0
            self.cache_read_tokens += getattr(usage, "cache_read_input_tokens", 0) or 0
            self.cache_write_tokens += getattr(usage, "cache_creation_input_tokens", 0) or 0

    def as_dict(self) -> dict[str, int]:
        return {
            "calls": self.calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_read_tokens": self.cache_read_tokens,
            "cache_write_tokens": self.cache_write_tokens,
        }


def strict_json_schema(schema: dict[str, Any] | type[BaseModel]) -> dict[str, Any]:
    """Return a copy of ``schema`` that strict tools / structured outputs accept.

    Every object gets ``additionalProperties: false`` and unsupported numeric /
    string constraints are removed (callers validate those client-side).
    """
    if isinstance(schema, type) and issubclass(schema, BaseModel):
        schema = schema.model_json_schema()
    schema = copy.deepcopy(schema)

    def walk(node: Any) -> Any:
        if isinstance(node, dict):
            for key in list(node):
                if key in _UNSUPPORTED_SCHEMA_KEYS and not isinstance(node[key], dict):
                    del node[key]
            if node.get("type") == "object" or "properties" in node:
                node["additionalProperties"] = False
            for value in node.values():
                walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)
        return node

    return walk(schema)


def response_text(response: Any) -> str:
    return "".join(
        getattr(block, "text", "") for block in response.content if getattr(block, "type", None) == "text"
    ).strip()


class ClaudeClient:
    """Synchronous Claude client with project defaults baked in."""

    def __init__(self, settings: Settings, client: anthropic.Anthropic | None = None):
        self.settings = settings
        # The SDK already retries 408/409/429/5xx and connection errors with
        # exponential backoff; we only raise the attempt count.
        self._client = client or anthropic.Anthropic(max_retries=4)
        self.usage = UsageTracker()

    # -- core request -------------------------------------------------------
    def _request_kwargs(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None,
        tools: list[dict[str, Any]] | None,
        effort: str | None,
        max_tokens: int | None,
        stop_sequences: list[str] | None,
        cache: bool,
    ) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "model": self.settings.model,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "messages": messages,
            "output_config": {"effort": effort or self.settings.effort},
        }
        if system:
            kwargs["system"] = system
        if tools:
            kwargs["tools"] = tools
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        if cache:
            # Auto-caches the last cacheable block: agent system prompts and
            # tool definitions are stable across the many calls of a run.
            kwargs["cache_control"] = {"type": "ephemeral"}
        if self.settings.use_fallbacks:
            kwargs["betas"] = [FALLBACK_BETA]
            kwargs["fallbacks"] = "default"
        return kwargs

    def create(
        self,
        *,
        messages: list[dict[str, Any]],
        system: str | list[dict[str, Any]] | None = None,
        tools: list[dict[str, Any]] | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
        stop_sequences: list[str] | None = None,
        cache: bool = True,
    ) -> Any:
        kwargs = self._request_kwargs(
            messages=messages, system=system, tools=tools, effort=effort,
            max_tokens=max_tokens, stop_sequences=stop_sequences, cache=cache,
        )
        api = self._client.beta.messages if self.settings.use_fallbacks else self._client.messages
        response = api.create(**kwargs)
        self.usage.add(getattr(response, "usage", None))
        self._raise_on_refusal(response)
        if response.stop_reason == "max_tokens":
            log.warning("response truncated at max_tokens=%s", kwargs["max_tokens"])
        return response

    def complete(self, prompt: str, *, system: str | None = None, effort: str | None = None) -> str:
        response = self.create(messages=[{"role": "user", "content": prompt}], system=system, effort=effort)
        return response_text(response)

    # -- structured outputs -------------------------------------------------
    def structured(
        self,
        schema: type[T],
        *,
        prompt: str,
        system: str | None = None,
        effort: str | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Return a validated ``schema`` instance using JSON structured outputs."""
        kwargs = self._request_kwargs(
            messages=[{"role": "user", "content": prompt}], system=system, tools=None,
            effort=effort, max_tokens=max_tokens, stop_sequences=None, cache=True,
        )
        api = self._client.beta.messages if self.settings.use_fallbacks else self._client.messages
        response = api.parse(output_format=schema, **kwargs)
        self.usage.add(getattr(response, "usage", None))
        self._raise_on_refusal(response)
        if response.stop_reason == "max_tokens":
            raise StructuredOutputError(f"{schema.__name__} output truncated at max_tokens")
        parsed = getattr(response, "parsed_output", None)
        if parsed is None:
            raise StructuredOutputError(f"no parsed {schema.__name__} in response")
        return parsed

    @staticmethod
    def _raise_on_refusal(response: Any) -> None:
        if getattr(response, "stop_reason", None) == "refusal":
            details = getattr(response, "stop_details", None)
            raise LLMRefusalError(
                getattr(details, "category", None), getattr(details, "explanation", None)
            )
