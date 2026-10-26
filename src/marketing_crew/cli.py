"""Command-line entry point: ``marketing-crew <command>`` or ``python -m marketing_crew``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
import uuid
from pathlib import Path

from marketing_crew.settings import PROJECT_ROOT, Settings

DEFAULT_BRIEF = PROJECT_ROOT / "examples" / "brief.json"


def _settings(args: argparse.Namespace) -> Settings:
    overrides = {}
    if getattr(args, "offline", False):
        overrides["offline"] = True
    if getattr(args, "redis_url", None):
        overrides["redis_url"] = args.redis_url
    return Settings.from_env(**overrides)


def _brief(path: str | None):
    from marketing_crew.workflow import CampaignBrief

    return CampaignBrief.model_validate_json(Path(path or DEFAULT_BRIEF).read_text())


def cmd_run(args: argparse.Namespace) -> int:
    from marketing_crew.workflow import MarketingWorkflow

    result = MarketingWorkflow(_settings(args)).run(
        _brief(args.brief), run_id=args.run_id, resume=not args.no_resume, verbose=args.verbose
    )
    print(result.final_output)
    print("\n" + "=" * 72)
    print(json.dumps({
        "run_id": result.run_id,
        "resumed_tasks": result.resumed_tasks,
        "verification": result.verification,
        "usage": result.usage,
        "rbac_denials": result.rbac_denials,
        "durable_memory": result.durable_memory,
        "artifacts": result.artifacts,
        "duration_s": result.duration_s,
    }, indent=2))
    return 0


def cmd_search(args: argparse.Namespace) -> int:
    from marketing_crew.runtime import build_services

    services = build_services(_settings(args), run_id=f"search-{uuid.uuid4().hex[:8]}")
    run = services.search_agent.run(args.question)
    print(json.dumps(json.loads(run.to_json()), indent=2))
    print("\nTrace:")
    for record in run.trace:
        status = "ok " if record.ok else "ERR"
        print(f"  step {record.step} {status} {record.tool} {json.dumps(record.args)}"
              + (f" -> {record.error}" if record.error else f" -> {record.evidence_id}"))
    return 0


def cmd_submit(args: argparse.Namespace) -> int:
    from marketing_crew.memory.store import connect_redis
    from marketing_crew.workflow import JobQueue

    settings = _settings(args)
    client, durable = connect_redis(settings.redis_url, allow_fallback=False)
    job_id = JobQueue(client, max_attempts=settings.job_max_attempts).submit(_brief(args.brief))
    print(job_id)
    return 0


def cmd_worker(args: argparse.Namespace) -> int:
    from marketing_crew.memory.store import connect_redis
    from marketing_crew.workflow import JobQueue, MarketingWorkflow

    settings = _settings(args)
    client, _ = connect_redis(settings.redis_url, allow_fallback=False)
    queue = JobQueue(client, max_attempts=settings.job_max_attempts)
    workflow = MarketingWorkflow(settings, redis_client=client)
    consumer = args.name or f"worker-{uuid.uuid4().hex[:6]}"
    print(f"{consumer} waiting for jobs (Ctrl+C to stop)")
    queue.work(workflow, consumer=consumer, once=args.once,
               on_result=lambda r: print(f"completed {r.run_id} -> {settings.output_dir / r.run_id}"))
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    from marketing_crew.memory.store import connect_redis
    from marketing_crew.workflow import JobQueue

    settings = _settings(args)
    client, _ = connect_redis(settings.redis_url, allow_fallback=False)
    queue = JobQueue(client)
    if args.dlq:
        for message in queue.dead_letters():
            print(message.model_dump_json())
    else:
        print(json.dumps(queue.status(args.job_id), indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="marketing-crew", description="Autonomous multi-agent marketing strategy system")
    parser.add_argument("--offline", action="store_true", help="scripted LLM + synthetic data; no network or API key")
    parser.add_argument("--redis-url", help="override REDIS_URL (use fakeredis:// for an in-process store)")
    parser.add_argument("-v", "--log-level", default="WARNING")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full crew on a brief")
    run.add_argument("--brief", help=f"brief JSON (default {DEFAULT_BRIEF.relative_to(PROJECT_ROOT)})")
    run.add_argument("--run-id", help="reuse an id to resume from its checkpoints")
    run.add_argument("--no-resume", action="store_true", help="ignore existing checkpoints")
    run.add_argument("--verbose", action="store_true", help="print CrewAI agent traces")
    run.set_defaults(func=cmd_run)

    search = sub.add_parser("search", help="run only the local search agent")
    search.add_argument("question")
    search.set_defaults(func=cmd_search)

    submit = sub.add_parser("submit", help="queue a brief for a worker (needs a real Redis)")
    submit.add_argument("--brief")
    submit.set_defaults(func=cmd_submit)

    worker = sub.add_parser("worker", help="process queued briefs")
    worker.add_argument("--name")
    worker.add_argument("--once", action="store_true", help="process available jobs, then exit")
    worker.set_defaults(func=cmd_worker)

    status = sub.add_parser("status", help="show a job's status or the dead-letter queue")
    status.add_argument("job_id", nargs="?")
    status.add_argument("--dlq", action="store_true")
    status.set_defaults(func=cmd_status)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=args.log_level.upper(), format="%(levelname)s %(name)s: %(message)s")
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
