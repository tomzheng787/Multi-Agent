"""CrewAI ``BaseLLM`` implementation backed by :class:`ClaudeClient`.

CrewAI agents run on the ReAct text protocol here (``Thought / Action /
Observation``), so ``supports_function_calling`` is False: CrewAI's native
tool path rebuilds the transcript in its own format, which would drop the
thinking blocks Claude needs to see again on tool turns. Native Claude
function calling is used where we own the loop - see
:mod:`marketing_crew.tools.local_search_agent`.
"""

from __future__ import annotations

from typing import Any

from crewai import BaseLLM
from pydantic import ConfigDict, Field, PrivateAttr

from marketing_crew.llm.claude import ClaudeClient, response_text

REACT_STYLE_NOTE = (
    "Keep each 'Thought:' line to one short sentence about your next step; put the "
    "substance of your work in the Action Input or the Final Answer."
)


def to_anthropic_messages(
    messages: str | list[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Convert CrewAI/OpenAI-style chat messages into (system, messages).

    System messages are hoisted into the top-level ``system`` field, consecutive
    same-role turns are merged, and the transcript is made to start with a user
    turn as the Messages API requires.
    """
    if isinstance(messages, str):
        return None, [{"role": "user", "content": messages}]

    system_parts: list[str] = []
    turns: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role", "user")
        content = message.get("content") or ""
        if isinstance(content, list):  # multimodal parts -> keep the text only
            content = "\n".join(p.get("text", "") for p in content if isinstance(p, dict))
        if role == "system":
            system_parts.append(content)
            continue
        role = "assistant" if role == "assistant" else "user"
        if turns and turns[-1]["role"] == role:
            turns[-1]["content"] += "\n\n" + content
        else:
            turns.append({"role": role, "content": content})

    if not turns or turns[0]["role"] != "user":
        turns.insert(0, {"role": "user", "content": "Begin."})
    # The API rejects assistant-prefill on current models; end on a user turn.
    if turns[-1]["role"] == "assistant":
        turns.append({"role": "user", "content": "Continue."})
    return ("\n\n".join(system_parts) or None), turns


class ClaudeCrewLLM(BaseLLM):
    """Lets CrewAI agents talk to Claude through the official Anthropic SDK."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    llm_type: str = "claude"
    provider: str = Field(default="anthropic")
    effort: str = "medium"
    _client: ClaudeClient = PrivateAttr()

    def __init__(self, client: ClaudeClient, *, effort: str | None = None, **kwargs: Any):
        super().__init__(model=client.settings.model, effort=effort or client.settings.effort, **kwargs)
        self._client = client

    def call(
        self,
        messages: str | list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        callbacks: list[Any] | None = None,
        available_functions: dict[str, Any] | None = None,
        from_task: Any | None = None,
        from_agent: Any | None = None,
        response_model: Any | None = None,
    ) -> str:
        system, turns = to_anthropic_messages(messages)
        system = f"{system}\n\n{REACT_STYLE_NOTE}" if system else REACT_STYLE_NOTE
        response = self._client.create(
            messages=turns,
            system=system,
            effort=self.effort,
            stop_sequences=self.stop_sequences or None,
        )
        return self._apply_stop_words(response_text(response))

    def supports_function_calling(self) -> bool:
        return False

    def supports_stop_words(self) -> bool:
        return True

    def get_context_window_size(self) -> int:
        # Leave CrewAI's summariser plenty of headroom inside the 1M window.
        return 800_000
