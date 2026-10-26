"""Runtime configuration, resolved from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = Path(os.getenv("MARKETING_CREW_CONFIG_DIR", PROJECT_ROOT / "config"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    # LLM
    model: str = "claude-opus-5-5"
    effort: str = "medium"  # Opus 5.5 defaults to medium; set explicitly
    manager_effort: str = "high"
    verifier_effort: str = "high"
    max_tokens: int = 16000
    use_fallbacks: bool = True  # server-side refusal fallbacks

    # Infrastructure
    redis_url: str = "redis://localhost:6379/0"
    redis_fallback: bool = True  # degrade to in-process store if Redis is down
    memory_ttl_seconds: int = 60 * 60 * 24 * 30
    memory_max_messages: int = 200

    # Behaviour
    offline: bool = False  # scripted LLM + canned HTTP; no network, no API key
    max_hallucination_rate: float = 0.10
    guardrail_max_retries: int = 2
    job_max_attempts: int = 3
    search_max_steps: int = 12
    http_user_agent: str = "marketing-crew/0.1 (multi-agent marketing research)"

    output_dir: Path = field(default_factory=lambda: PROJECT_ROOT / "outputs")
    config_dir: Path = CONFIG_DIR

    @classmethod
    def from_env(cls, **overrides) -> "Settings":
        values = dict(
            model=os.getenv("MC_MODEL", cls.model),
            effort=os.getenv("MC_EFFORT", cls.effort),
            manager_effort=os.getenv("MC_MANAGER_EFFORT", cls.manager_effort),
            verifier_effort=os.getenv("MC_VERIFIER_EFFORT", cls.verifier_effort),
            max_tokens=int(os.getenv("MC_MAX_TOKENS", cls.max_tokens)),
            use_fallbacks=_env_bool("MC_USE_FALLBACKS", cls.use_fallbacks),
            redis_url=os.getenv("REDIS_URL", cls.redis_url),
            redis_fallback=_env_bool("MC_REDIS_FALLBACK", cls.redis_fallback),
            memory_ttl_seconds=int(os.getenv("MC_MEMORY_TTL", cls.memory_ttl_seconds)),
            offline=_env_bool("MC_OFFLINE", cls.offline),
            max_hallucination_rate=float(
                os.getenv("MC_MAX_HALLUCINATION_RATE", cls.max_hallucination_rate)
            ),
            guardrail_max_retries=int(
                os.getenv("MC_GUARDRAIL_RETRIES", cls.guardrail_max_retries)
            ),
            http_user_agent=os.getenv("MC_HTTP_USER_AGENT", cls.http_user_agent),
            output_dir=Path(os.getenv("MC_OUTPUT_DIR", PROJECT_ROOT / "outputs")),
        )
        values.update(overrides)
        return cls(**values)
