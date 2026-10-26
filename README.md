# Marketing Crew

An autonomous multi-agent system that turns a campaign brief into a verified, execution-ready
marketing strategy. It's built on **CrewAI** (hierarchical crew), **LangChain** (tools, chat
memory) and **Claude** (through the official Anthropic SDK).

```
brief ─▶ Chief Marketing Officer (manager)
           │ delegates (RBAC-checked)
           ├─▶ Market Researcher ──▶ local search agent ──▶ live APIs (OSM, Wikipedia)
           ├─▶ Market Analyst                    │ every observation stored as evidence [E#]
           ├─▶ Strategy Writer                   ▼
           ├─▶ Strategy Critic  ◀── chain-of-verification guardrail on each deliverable
           └─▶ Campaign Executor ─▶ budget split, schedule, saved artifacts
         Redis: conversation memory · evidence · checkpoints · message bus · job queue
```

## How the requirements map to the code

| Requirement | Where |
|---|---|
| 5+ specialised agents (researcher, analyst, writer, critic, executor) + manager | [config/agents.yaml](config/agents.yaml), [crew/builder.py](src/marketing_crew/crew/builder.py) |
| Hierarchical task delegation | `Process.hierarchical` with a tool-less manager; each task is pinned to its owner ([config/tasks.yaml](config/tasks.yaml)) |
| Role-based access control | [config/rbac.yaml](config/rbac.yaml), [security/rbac.py](src/marketing_crew/security/rbac.py): least-privilege tool grants at build time, a check on every tool call, a `before_tool_call` hook on delegation, and an audit log |
| End-to-end marketing strategy automation | research → analysis → strategy → review → execution (budget split, content schedule, saved `strategy.md` / `launch_plan.md`) |
| Tool-augmented local search agent | [tools/local_search_agent.py](src/marketing_crew/tools/local_search_agent.py): native Claude function calling over live OpenStreetMap / Wikipedia data |
| Dynamic API invocation | [config/apis.yaml](config/apis.yaml) → [tools/api_registry.py](src/marketing_crew/tools/api_registry.py): APIs are declared, not coded; each one becomes a strict Claude tool |
| Self-corrective retry logic | Backoff + `Retry-After`, circuit breakers, schema validation with hints, error results the model repairs, a submit tool that rejects made-up evidence ids, nudges, and a step budget with a partial report |
| Persistent Redis-backed memory | [memory/store.py](src/marketing_crew/memory/store.py): LangChain `BaseChatMessageHistory` on Redis (session and per-agent), with TTL, trimming, write buffering and an in-process fallback |
| Inter-agent messaging over a queue | [messaging/bus.py](src/marketing_crew/messaging/bus.py): Redis Streams, consumer groups, acks, retries, a dead-letter stream and `XAUTOCLAIM` recovery; agents use `send_agent_message` / `read_agent_inbox` |
| Fault-tolerant workflows | [workflow.py](src/marketing_crew/workflow.py): per-task checkpoints with resume, a durable job queue with retry and DLQ, and output-integrity gates |
| Chain-of-thought verification to reduce hallucination | [verification/cove.py](src/marketing_crew/verification/cove.py): the deliverable is broken into a claim chain, then checked by deterministic checks, a factored LLM judge and premise propagation. It runs as a CrewAI guardrail that sends unsupported claims back for revision |
| Measuring the reduction | [eval/hallucination_eval.py](eval/hallucination_eval.py): A/B test with an independent judge (see [Measuring hallucination](#measuring-hallucination)) |

## Quick start

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

**Offline (no API key, no network, no Redis).** The LLM is scripted and the data is synthetic. Use
it to see the whole system run:

```bash
marketing-crew --offline --redis-url fakeredis:// run
```

The offline script exercises the failure paths on purpose. The search agent first sends an
out-of-range radius and repairs it from the error hint. The writer's first draft cites a made-up
statistic; verification rejects it and the revised draft passes. The run report shows both.

**Live run (Claude + public APIs + Redis):**

```bash
docker compose up -d redis
cp .env.example .env   # set ANTHROPIC_API_KEY (or use `ant auth login`) and MC_HTTP_USER_AGENT
set -a; source .env; set +a
marketing-crew run --brief examples/brief.json
```

Outputs go to `outputs/<run_id>/`: `final_output.md`, `strategy.md`, `launch_plan.md`,
`campaign_schedule.json` and `run_report.json`. The run report holds verification per task, token
usage, RBAC denials and traces of each search.

**Other commands**

```bash
marketing-crew search "Which gyms compete with a boutique studio in Shoreditch, London?"
marketing-crew run --run-id run-abc123      # resume a failed run from its checkpoints
marketing-crew submit --brief examples/brief.json   # queue a brief (needs a real Redis)
marketing-crew worker                        # process queued briefs (run several for parallelism)
marketing-crew status run-abc123             # job state; add --dlq to list dead-lettered jobs
```

## Design notes

**LLM layer.** Every agent calls Claude through [`ClaudeClient`](src/marketing_crew/llm/claude.py),
which uses the official Anthropic SDK. It defaults to `claude-opus-5-5` with an explicit `effort`
(the manager, critic and verifier run at `high`), prompt caching, and structured outputs for
verification. **Server-side refusal fallbacks are on by default** (`fallbacks: "default"`): if
Claude declines a request, the API retries it on Anthropic's recommended fallback model. Turn this
off with `MC_USE_FALLBACKS=false`. A refusal that survives the fallback raises `LLMRefusalError`.
CrewAI agents use the ReAct text protocol through
[`ClaudeCrewLLM`](src/marketing_crew/llm/crewai_adapter.py). The search agent uses native
function calling, because there we own the loop and can pass thinking blocks back unchanged.

**RBAC.** Roles inherit permissions and support wildcards. Each tool maps to one permission;
unknown tools are denied. Agents only receive tools their role allows. Each tool call is checked
again at run time, and a CrewAI hook blocks delegation the policy doesn't allow (for example a
worker delegating, or the manager delegating to itself). Denials are audited and published to the
event stream.

**Verification protocol.** Asking a model to write out its hidden reasoning is unreliable and can
trigger refusals. So the protocol checks the reasoning chain in the deliverable instead:

1. Extract claims. Each is labelled factual, numeric, inference or recommendation, with the evidence ids it cites and the claims it depends on.
2. Run deterministic checks: cited evidence must exist, and every figure must appear in the cited evidence or in the brief.
3. Judge each remaining claim on its own against its evidence. This step never sees the draft (factored verification).
4. Propagate failures: an inference or recommendation that rests on a flagged premise is flagged too.

The hallucination rate is the share of checkable claims that get flagged. Above the threshold
(default 10%) or on any contradiction, the guardrail returns targeted feedback and the task is
retried. If claims are still flagged on the last retry, the output is accepted with the findings
attached as a visible warning. Offline, a lexical matcher stands in for the LLM judge.

**Fault tolerance.**
- **Redis:** transient errors are retried, and memory writes are buffered until Redis recovers. If Redis is unreachable at start-up, the run falls back to an in-process store and `durable_memory: false` appears in the report.
- **HTTP:** retries with backoff that honour `Retry-After`, plus per-API circuit breakers and per-API rate limits (Nominatim: 1 request per second).
- **Tasks:** checkpointed when they finish, so a crashed run resumes from the last finished task. An integrity gate rejects empty or failed outputs, such as a worker's crash coming back as text through the delegation tool.
- **Jobs:** delivered at least once. Failures are retried (resuming from checkpoints) and dead-lettered after the attempt budget. Jobs held by crashed workers are reclaimed.

## Measuring hallucination

The verification protocol is built to cut hallucination. **The size of the cut depends on your
data and has to be measured, not assumed.** The eval harness runs an A/B test on each case: a
draft with no protocol versus the same draft after verify → revise → re-verify. An independent
judge scores both drafts.

```bash
python eval/hallucination_eval.py            # dry run: prints the planned number of API calls
python eval/hallucination_eval.py --yes      # runs it; writes eval/results/run-*.json
```

It reports the mean hallucination rate for both arms and the relative reduction. The bundled
cases use synthetic evidence. Add cases from your own domain, and confirm the result on a
human-labelled sample before you quote a number: the judge is the same model family as the
verifier. Each live crew run also records first-attempt versus final rates per task in
`run_report.json`.

## Tests

```bash
pytest
```

51 tests run fully offline using fakeredis and mock HTTP. They cover RBAC, memory resilience, the
bus (retry, DLQ, reclaim), API retries and circuit breakers, the search agent's self-correction,
verification layers and propagation, the Claude request shape, end-to-end crew runs, crash and
resume, and the job queue's retry and dead-lettering.

## Extending

- **New data source:** add an entry to `config/apis.yaml`. Use `kind: http` for declarative calls, or `kind: python` pointing at a function that receives a resilient `fetch`.
- **New agent or tool:** add a LangChain tool in `tools/toolkit.py`, map it to a permission in `rbac.yaml`, and list it under the agent in `agents.yaml`.
- **Real channel publishing:** `schedule_campaign` only writes a local file. Connect your scheduler there, behind the `channel:schedule` permission.
