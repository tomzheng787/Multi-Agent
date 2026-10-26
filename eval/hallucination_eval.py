"""A/B evaluation: hallucination rate with vs. without chain-of-verification.

For each case the same writer prompt produces a draft (arm A, no protocol).
Arm B takes that draft through the verification protocol - verify, revise with
the findings, re-verify - exactly as the task guardrail does inside the crew.
An *independent* judge (joint, whole-document prompt; not the verifier's
claim-by-claim pipeline) then scores every factual claim of both drafts
against the case evidence.

    hallucination rate = unsupported factual claims / factual claims
    relative reduction = (rate_A - rate_B) / rate_A

This calls the live API (it costs money) and needs --yes to run. The judge is
the same model family as the verifier, so its errors may correlate with the
verifier's; confirm headline numbers on a human-labelled sample before quoting
them.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import fakeredis
import yaml
from pydantic import BaseModel

from marketing_crew.llm.claude import ClaudeClient
from marketing_crew.memory.store import EvidenceStore
from marketing_crew.settings import Settings
from marketing_crew.verification.cove import ChainOfVerification

HERE = Path(__file__).parent

WRITER_SYSTEM = (
    "You are a senior marketing strategist. Ground every market fact and figure in the evidence "
    "provided and cite its id like [E2] right after the claim. Budgets and targets are plans; phrase "
    "them as such."
)

JUDGE_SYSTEM = (
    "You audit marketing documents. List every factual claim about the world in the document "
    "(skip plans, targets, recommendations and opinions). For each, decide whether the evidence "
    "provided supports it. A claim with a figure that does not appear in the evidence is unsupported. "
    "Use only the evidence, not general knowledge."
)


class JudgedClaim(BaseModel):
    claim: str
    supported: bool


class Judgement(BaseModel):
    claims: list[JudgedClaim]

    @property
    def rate(self) -> float:
        return sum(not c.supported for c in self.claims) / len(self.claims) if self.claims else 0.0


def load_case(case: dict, run_id: str) -> EvidenceStore:
    store = EvidenceStore(fakeredis.FakeRedis(decode_responses=True), run_id)
    for item in case["evidence"]:
        store.add(source=item["source"], title=item.get("title"), content=item["content"])
    return store


def judge(llm: ClaudeClient, document: str, evidence: str) -> Judgement:
    prompt = f"<evidence>\n{evidence}\n</evidence>\n\n<document>\n{document}\n</document>"
    return llm.structured(Judgement, system=JUDGE_SYSTEM, prompt=prompt, effort="high")


def run_case(llm: ClaudeClient, case: dict, *, threshold: float, max_rounds: int) -> dict:
    store = load_case(case, f"eval-{case['id']}")
    evidence = store.render()
    draft = llm.complete(f"<evidence>\n{evidence}\n</evidence>\n\n{case['task']}", system=WRITER_SYSTEM)

    verifier = ChainOfVerification(store, llm, effort="high")
    verified, rounds = draft, 0
    report = verifier.verify(verified)
    while not report.passed(threshold) and rounds < max_rounds:
        verified = verifier.revise(verified, report)
        report = verifier.verify(verified)
        rounds += 1

    baseline_j, verified_j = judge(llm, draft, evidence), judge(llm, verified, evidence)
    return {
        "id": case["id"],
        "baseline_rate": round(baseline_j.rate, 4),
        "verified_rate": round(verified_j.rate, 4),
        "baseline_claims": len(baseline_j.claims),
        "verified_claims": len(verified_j.claims),
        "revision_rounds": rounds,
        "verifier_final": report.summary(),
        "baseline_unsupported": [c.claim for c in baseline_j.claims if not c.supported],
        "verified_unsupported": [c.claim for c in verified_j.claims if not c.supported],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cases", default=str(HERE / "cases.yaml"))
    parser.add_argument("--max-rounds", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--yes", action="store_true", help="confirm you want to spend API credits")
    args = parser.parse_args()

    cases = yaml.safe_load(Path(args.cases).read_text())["cases"]
    print(f"{len(cases)} cases; roughly {len(cases) * (4 + 10 * (args.max_rounds + 1))} API calls "
          f"(writer + claim extraction + per-claim verification + revisions + 2 judge calls per case).")
    if not args.yes:
        print("Dry run. Re-run with --yes to call the API.")
        return

    llm = ClaudeClient(Settings.from_env())
    results = []
    for case in cases:
        result = run_case(llm, case, threshold=args.threshold, max_rounds=args.max_rounds)
        results.append(result)
        print(f"{case['id']:<24} baseline {result['baseline_rate']:.0%}  verified {result['verified_rate']:.0%}")

    baseline = statistics.mean(r["baseline_rate"] for r in results)
    verified = statistics.mean(r["verified_rate"] for r in results)
    summary = {
        "cases": len(results),
        "mean_baseline_rate": round(baseline, 4),
        "mean_verified_rate": round(verified, 4),
        "relative_reduction": round((baseline - verified) / baseline, 4) if baseline else None,
        "usage": llm.usage.as_dict(),
    }
    print(json.dumps(summary, indent=2))
    out = HERE / "results" / f"run-{int(time.time())}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
