"""Dynamic API invocation for tool-using agents.

APIs are declared in ``config/apis.yaml`` rather than hard-coded: each spec is
turned into a Claude tool definition automatically and invoked through one
resilient executor. Adding a data source is a config change.

Resilience layers, from the wire up:

1. transport retries with exponential backoff + jitter, honouring
   ``Retry-After`` on 429/503;
2. a per-API circuit breaker so a dead upstream fails fast;
3. argument validation against the spec's JSON schema *before* any network
   call, returning precise errors;
4. every failure (and every empty result) comes back with a ``hint`` so the
   calling model can correct its arguments and try again.
"""

from __future__ import annotations

import importlib
import json
import logging
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import httpx
import yaml
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field
from tenacity import (
    RetryError,
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential_jitter,
)

from marketing_crew.llm.claude import strict_json_schema

log = logging.getLogger(__name__)

_PLACEHOLDER = re.compile(r"\{(\w+)\}")
_TAG = re.compile(r"<[^>]+>")


class ToolInputError(ValueError):
    """Raised by API functions for bad arguments the model can fix."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.hint = hint


class RetryableHTTPError(Exception):
    def __init__(self, status: int, url: str, retry_after: float | None = None):
        super().__init__(f"HTTP {status} from {url}")
        self.status = status
        self.retry_after = retry_after


class CircuitOpenError(Exception):
    pass


class ApiSpec(BaseModel):
    name: str
    description: str
    kind: str = "http"  # "http" | "python"
    method: str = "GET"
    url: str | None = None
    query: dict[str, Any] = Field(default_factory=dict)
    headers: dict[str, str] = Field(default_factory=dict)
    function: str | None = None  # "package.module:callable" for kind=python
    params_schema: dict[str, Any]
    defaults: dict[str, Any] = Field(default_factory=dict)
    result_path: str | None = None
    result_fields: dict[str, str] | None = None
    max_items: int = 10
    strip_html: bool = False
    empty_hint: str = "No results. Broaden or rephrase the query and try again."
    min_interval_s: float = 0.0
    timeout_s: float = 20.0

    def tool_definition(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description.strip(),
            "input_schema": strict_json_schema(self.params_schema),
            "strict": True,
        }


@dataclass
class ToolOutcome:
    ok: bool
    data: Any = None
    error: str | None = None
    hint: str | None = None
    source_urls: list[str] = field(default_factory=list)
    latency_ms: float = 0.0

    @property
    def empty(self) -> bool:
        return self.ok and (self.data in (None, [], {}, ""))

    def to_payload(self, **extra: Any) -> dict[str, Any]:
        payload: dict[str, Any] = {"ok": self.ok, **extra}
        if self.ok:
            payload["data"] = self.data
        else:
            payload["error"] = self.error
        if self.hint:
            payload["hint"] = self.hint
        return payload


class CircuitBreaker:
    """Opens after ``threshold`` consecutive failures; half-opens after ``reset_s``."""

    def __init__(self, threshold: int = 3, reset_s: float = 30.0, clock: Callable[[], float] = time.monotonic):
        self.threshold = threshold
        self.reset_s = reset_s
        self.clock = clock
        self.failures = 0
        self.opened_at: float | None = None
        self._lock = threading.Lock()

    @property
    def state(self) -> str:
        if self.opened_at is None:
            return "closed"
        return "half_open" if self.clock() - self.opened_at >= self.reset_s else "open"

    def before_call(self) -> None:
        if self.state == "open":
            raise CircuitOpenError("circuit open")

    def record_success(self) -> None:
        with self._lock:
            self.failures = 0
            self.opened_at = None

    def record_failure(self) -> None:
        with self._lock:
            self.failures += 1
            if self.failures >= self.threshold or self.opened_at is not None:
                self.opened_at = self.clock()


class HttpFetcher:
    """JSON-over-HTTP with retry/backoff. Returns parsed JSON (or text)."""

    RETRYABLE_STATUS = {408, 425, 429, 500, 502, 503, 504}

    def __init__(
        self,
        client: httpx.Client | None = None,
        *,
        user_agent: str,
        max_attempts: int = 3,
        base_delay_s: float = 0.5,
        max_delay_s: float = 8.0,
    ):
        self.client = client or httpx.Client(follow_redirects=True)
        self.user_agent = user_agent
        self.max_attempts = max_attempts
        self.base_delay_s = base_delay_s
        self.max_delay_s = max_delay_s

    def _wait(self, retry_state: Any) -> float:
        exc = retry_state.outcome.exception() if retry_state.outcome else None
        if isinstance(exc, RetryableHTTPError) and exc.retry_after is not None:
            return min(exc.retry_after, self.max_delay_s)
        return wait_exponential_jitter(initial=self.base_delay_s, max=self.max_delay_s)(retry_state)

    def __call__(
        self,
        method: str,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 20.0,
        on_request: Callable[[str], None] | None = None,
    ) -> Any:
        merged_headers = {"User-Agent": self.user_agent, "Accept": "application/json", **(headers or {})}

        def attempt() -> Any:
            response = self.client.request(
                method, url, params=params, data=data, headers=merged_headers, timeout=timeout
            )
            if on_request:
                on_request(str(response.request.url))
            if response.status_code in self.RETRYABLE_STATUS:
                retry_after = response.headers.get("retry-after")
                raise RetryableHTTPError(
                    response.status_code, url, float(retry_after) if retry_after and retry_after.isdigit() else None
                )
            response.raise_for_status()
            try:
                return response.json()
            except ValueError:
                return response.text

        retrying = Retrying(
            retry=retry_if_exception_type((RetryableHTTPError, httpx.TransportError)),
            stop=stop_after_attempt(self.max_attempts),
            wait=self._wait,
            reraise=True,
        )
        return retrying(attempt)


def _dig(data: Any, path: str | None) -> Any:
    if not path:
        return data
    for part in path.split("."):
        if isinstance(data, dict):
            data = data.get(part)
        elif isinstance(data, list) and part.isdigit():
            data = data[int(part)] if int(part) < len(data) else None
        else:
            return None
    return data


def _clean(value: Any, strip_html: bool, max_chars: int = 1500) -> Any:
    if isinstance(value, str):
        if strip_html:
            value = _TAG.sub("", value)
        return value if len(value) <= max_chars else value[: max_chars - 3] + "..."
    if isinstance(value, list):
        return [_clean(v, strip_html, max_chars) for v in value]
    if isinstance(value, dict):
        return {k: _clean(v, strip_html, max_chars) for k, v in value.items()}
    return value


class ApiRegistry:
    def __init__(self, specs: list[ApiSpec]):
        self.specs = {spec.name: spec for spec in specs}
        self._validators = {spec.name: Draft202012Validator(spec.params_schema) for spec in specs}

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ApiRegistry":
        data = yaml.safe_load(Path(path).read_text())
        return cls([ApiSpec.model_validate(item) for item in data.get("apis", [])])

    def tool_definitions(self, names: list[str] | None = None) -> list[dict[str, Any]]:
        selected = names or list(self.specs)
        return [self.specs[name].tool_definition() for name in selected]

    def validate(self, name: str, args: dict[str, Any]) -> list[str]:
        return [
            f"{'/'.join(map(str, err.path)) or 'arguments'}: {err.message}"
            for err in self._validators[name].iter_errors(args)
        ]


class ApiInvoker:
    """Executes registry specs by name with validation, retries and breakers."""

    def __init__(self, registry: ApiRegistry, fetcher: HttpFetcher, *, breaker_threshold: int = 3, breaker_reset_s: float = 30.0):
        self.registry = registry
        self.fetcher = fetcher
        self.breakers = {name: CircuitBreaker(breaker_threshold, breaker_reset_s) for name in registry.specs}
        self._last_call: dict[str, float] = {}
        self._rate_locks = {name: threading.Lock() for name in registry.specs}

    def invoke(self, name: str, args: dict[str, Any] | None) -> ToolOutcome:
        started = time.perf_counter()
        outcome = self._invoke(name, dict(args or {}))
        outcome.latency_ms = round((time.perf_counter() - started) * 1000, 1)
        return outcome

    def _invoke(self, name: str, args: dict[str, Any]) -> ToolOutcome:
        spec = self.registry.specs.get(name)
        if spec is None:
            return ToolOutcome(
                ok=False, error=f"unknown tool '{name}'",
                hint=f"Available tools: {', '.join(sorted(self.registry.specs))}",
            )
        args = {**spec.defaults, **{k: v for k, v in args.items() if v is not None}}
        problems = self.registry.validate(name, args)
        if problems:
            return ToolOutcome(
                ok=False, error="invalid arguments: " + "; ".join(problems),
                hint="Fix the listed arguments to match the tool's input schema and call it again.",
            )

        breaker = self.breakers[name]
        try:
            breaker.before_call()
        except CircuitOpenError:
            return ToolOutcome(
                ok=False, error=f"'{name}' is temporarily unavailable (circuit open after repeated failures)",
                hint="Use a different tool or source for this information, or record it as a data gap.",
            )

        urls: list[str] = []
        try:
            self._respect_rate_limit(spec)
            raw = self._execute(spec, args, urls)
        except ToolInputError as exc:
            # The upstream is healthy; the arguments were wrong.
            return ToolOutcome(ok=False, error=str(exc), hint=exc.hint, source_urls=urls)
        except (RetryableHTTPError, httpx.TransportError, RetryError) as exc:
            breaker.record_failure()
            return ToolOutcome(
                ok=False, error=f"upstream unavailable after {self.fetcher.max_attempts} attempts: {exc}",
                hint="Try again with different parameters later, use another tool, or note this as a data gap.",
                source_urls=urls,
            )
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status >= 500:
                breaker.record_failure()
            return ToolOutcome(
                ok=False, error=f"HTTP {status}: {exc.response.text[:300]}",
                hint=spec.empty_hint if status == 404
                else "The request was rejected; check spelling and parameter values, then retry.",
                source_urls=urls,
            )
        breaker.record_success()

        data = self._extract(spec, raw)
        outcome = ToolOutcome(ok=True, data=data, source_urls=urls)
        if outcome.empty:
            outcome.hint = spec.empty_hint
        return outcome

    def _respect_rate_limit(self, spec: ApiSpec) -> None:
        if spec.min_interval_s <= 0:
            return
        with self._rate_locks[spec.name]:
            wait = spec.min_interval_s - (time.monotonic() - self._last_call.get(spec.name, 0.0))
            if wait > 0:
                time.sleep(wait)
            self._last_call[spec.name] = time.monotonic()

    def _execute(self, spec: ApiSpec, args: dict[str, Any], urls: list[str]) -> Any:
        if spec.kind == "python":
            module_name, _, attr = (spec.function or "").partition(":")
            func = getattr(importlib.import_module(module_name), attr)

            def fetch(method: str, url: str, **kwargs: Any) -> Any:
                return self.fetcher(method, url, timeout=spec.timeout_s, on_request=urls.append, **kwargs)

            return func(fetch, **args)

        url = _PLACEHOLDER.sub(lambda m: quote(str(args[m.group(1)]), safe=""), spec.url or "")
        params: dict[str, Any] = {}
        for key, template in spec.query.items():
            if isinstance(template, str) and _PLACEHOLDER.fullmatch(template):
                value = args.get(template[1:-1])
                if value is not None:
                    params[key] = value
            else:
                params[key] = template
        return self.fetcher(
            spec.method, url, params=params, headers=spec.headers,
            timeout=spec.timeout_s, on_request=urls.append,
        )

    @staticmethod
    def _extract(spec: ApiSpec, raw: Any) -> Any:
        data = _dig(raw, spec.result_path)
        if isinstance(data, dict) and spec.result_fields:
            data = {alias: _dig(data, path) for alias, path in spec.result_fields.items()}
        elif isinstance(data, list):
            data = data[: spec.max_items]
            if spec.result_fields:
                data = [
                    {alias: _dig(item, path) for alias, path in spec.result_fields.items()}
                    for item in data
                ]
        return _clean(data, spec.strip_html)


def outcome_to_json(outcome: ToolOutcome, **extra: Any) -> str:
    return json.dumps(outcome.to_payload(**extra), ensure_ascii=False, default=str)
