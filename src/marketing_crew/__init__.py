"""Autonomous multi-agent marketing strategy system (CrewAI + LangChain + Claude)."""

import os

# CrewAI phones home and may prompt about tracing by default; this system keeps
# all state in its own Redis layer, so switch both off unless explicitly set.
os.environ.setdefault("CREWAI_DISABLE_TELEMETRY", "true")
os.environ.setdefault("CREWAI_TRACING_ENABLED", "false")
os.environ.setdefault("OTEL_SDK_DISABLED", "true")

__version__ = "0.1.0"
