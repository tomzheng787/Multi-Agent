"""Chain-of-verification protocol for agent deliverables.

A deliverable is decomposed into atomic claims that form a reasoning chain:
facts and figures cite evidence ids, inferences and recommendations name the
claims they rest on. Verification then runs in three layers:

1. deterministic checks (no LLM): cited evidence must exist, and every figure
   in a factual claim must appear in its cited evidence or the brief;
2. factored LLM verification: each remaining claim is judged *in isolation*
   against its evidence - the verifier never sees the draft, which avoids
   rubber-stamping it;
3. chain propagation: an inference or recommendation is only as good as its
   premises; anything resting on a flagged premise is flagged too.

The share of flagged checkable claims is the hallucination rate. Wired in as
a CrewAI task guardrail, a failing report sends targeted feedback back to the
agent, which revises before the workflow moves on.
"""

from __future__ import annotations

import logging
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, Field

from marketing_crew.llm.claude import ClaudeClient, response_text
from marketing_crew.memory.store import Evidence, EvidenceStore, _tokens

log = logging.getLogger(__name__)

ClaimKind = Literal["factual", "numeric", "inference", "recommendation"]
Status = Literal["supported", "contradicted", "unsupported"]

FEEDBACK_MARKER = "VERIFICATION FAILED"

_CITATION_GROUP = re.compile(r"\[([^\]]*\bE\d+\b[^\]]*)\]")
_EVIDENCE_ID = re.compile(r"\bE\d+\b")
_NUMBER = re.compile(r"(?<![\w.])\d[\d,]*(?:\.\d+)?")
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\[\"(*])|\n+")
_RECOMMENDATION = re.compile(
    r"\b(should|recommend|we will|will launch|plan to|aim to|allocate|target(?:ing)?|launch|"
    r"prioriti[sz]e|invest|schedule|kpi|goal)\b",
    re.I,
)


class Claim(BaseModel):
    id: str
    text: str
    kind: ClaimKind
    cited_evidence: list[str] = Field(description="Evidence ids the document cites for this claim, e.g. E3")
    depends_on: list[str] = Field(description="Ids of earlier claims this claim is derived from")


class ClaimSet(BaseModel):
    claims: list[Claim]


class LLMVerdict(BaseModel):
    status: Status
    evidence_ids: list[str]
    rationale: str


class ClaimVerdict(BaseModel):
    claim_id: str
    status: Status
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = ""
    method: str = "llm"  # deterministic | lexical | llm | chain


class VerificationReport(BaseModel):
    claims: list[Claim]
    verdicts: list[ClaimVerdict]

    def verdict(self, claim_id: str) -> ClaimVerdict | None:
        return next((v for v in self.verdicts if v.claim_id == claim_id), None)

    @property
    def checkable(self) -> list[Claim]:
        # Recommendations are choices, not facts; they count only via their premises.
        return [c for c in self.claims if c.kind != "recommendation"]

    @property
    def flagged(self) -> list[ClaimVerdict]:
        return [v for v in self.verdicts if v.status != "supported"]

    @property
    def hallucination_rate(self) -> float:
        checkable = {c.id for c in self.checkable}
        if not checkable:
            return 0.0
        bad = sum(1 for v in self.flagged if v.claim_id in checkable)
        return bad / len(checkable)

    @property
    def contradictions(self) -> int:
        return sum(1 for v in self.verdicts if v.status == "contradicted")

    def passed(self, threshold: float) -> bool:
        return self.hallucination_rate <= threshold and self.contradictions == 0

    def feedback(self, threshold: float) -> str:
        claims = {c.id: c for c in self.claims}
        lines = [
            f"{FEEDBACK_MARKER} - {len(self.flagged)} claim(s) are not backed by the evidence store "
            f"(hallucination rate {self.hallucination_rate:.0%}, limit {threshold:.0%}).",
            "Fix every item below, then return the COMPLETE revised deliverable:",
        ]
        for v in self.flagged:
            text = claims[v.claim_id].text if v.claim_id in claims else ""
            lines.append(f'- [{v.claim_id}, {v.status}] "{_short(text, 220)}" - {v.rationale}')
        lines.append(
            "Rules: back each claim with an existing evidence id (use evidence_lookup to find one), "
            "restate it explicitly as an assumption, or remove it. Never invent evidence ids or figures."
        )
        return "\n".join(lines)

    def summary(self) -> dict[str, Any]:
        return {
            "claims": len(self.claims),
            "checkable": len(self.checkable),
            "flagged": len(self.flagged),
            "contradicted": self.contradictions,
            "hallucination_rate": round(self.hallucination_rate, 4),
        }


def _short(text: str, limit: int) -> str:
    text = " ".join(text.split())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _normalise_numbers(text: str) -> set[str]:
    text = _EVIDENCE_ID.sub(" ", text)
    out = set()
    for raw in _NUMBER.findall(text):
        value = raw.replace(",", "").rstrip(".")
        try:
            number = float(value)
        except ValueError:
            continue
        out.add(f"{number:g}")
    return out


def cited_ids(text: str) -> list[str]:
    ids: list[str] = []
    for group in _CITATION_GROUP.findall(text):
        for eid in _EVIDENCE_ID.findall(group):
            if eid not in ids:
                ids.append(eid)
    return ids


def heuristic_claims(text: str) -> list[Claim]:
    """Deterministic claim extraction used offline and as a fallback.

    Keeps sentences that cite evidence, contain figures or propose actions.
    """
    claims: list[Claim] = []
    for raw in _SENTENCE_SPLIT.split(text):
        sentence = re.sub(r"^[\s#>*\-\d.)]+", "", raw).strip(" *_`")
        if len(sentence) < 20 or sentence.endswith(":"):
            continue
        cites = cited_ids(sentence)
        body = _CITATION_GROUP.sub("", sentence)
        if _RECOMMENDATION.search(body):
            kind: ClaimKind = "recommendation"
        elif _normalise_numbers(body):
            kind = "numeric"
        elif cites:
            kind = "factual"
        else:
            continue
        claims.append(Claim(id=f"C{len(claims) + 1}", text=sentence, kind=kind, cited_evidence=cites, depends_on=[]))
    return claims


EXTRACT_SYSTEM = """\
You decompose marketing deliverables into atomic, independently checkable claims for a \
fact-checking pipeline. Claim kinds:
- factual: a statement about the world (market, competitors, place, customers).
- numeric: a factual statement containing a figure (count, percentage, price, distance, date).
- inference: a conclusion drawn from other claims; list those claim ids in depends_on.
- recommendation: a proposed action or planning target (budgets, KPIs, schedules). These are \
choices, not facts; list the claims they rely on in depends_on.
Copy the evidence ids the document cites for each claim (e.g. "E3") into cited_evidence. Number \
claims C1, C2, ... in document order; depends_on may only reference earlier claims. Skip headings \
and boilerplate."""

VERIFY_SYSTEM = """\
You are a strict fact-checker. Judge a single claim using ONLY the evidence provided - not general \
knowledge. supported: the evidence states it or directly entails it. contradicted: the evidence \
states otherwise. unsupported: the evidence does not establish it (including figures that do not \
appear in the evidence). For an inference, also judge whether it follows from the verified \
premises. Cite the evidence ids you relied on and give a one-sentence rationale."""

REVISE_SYSTEM = """\
You revise marketing deliverables after fact-checking. Remove, correct (using the evidence) or \
explicitly mark as an assumption every flagged claim. Keep everything else, including structure \
and citations, unchanged. Return only the revised document."""


class ChainOfVerification:
    def __init__(
        self,
        evidence: EvidenceStore,
        llm: ClaudeClient | None = None,
        *,
        effort: str = "high",
        top_k: int = 4,
        max_workers: int = 4,
        lexical_threshold: float = 0.3,
    ):
        self.evidence = evidence
        self.llm = llm
        self.effort = effort
        self.top_k = top_k
        self.max_workers = max_workers
        self.lexical_threshold = lexical_threshold

    # -- public API ---------------------------------------------------------
    def verify(self, text: str, *, allowed_facts: str = "") -> VerificationReport:
        claims = self.extract_claims(text)
        allowed_numbers = _normalise_numbers(allowed_facts)
        verdicts: dict[str, ClaimVerdict] = {}

        base = [c for c in claims if c.kind in ("factual", "numeric") or not c.depends_on]
        derived = [c for c in claims if c not in base]

        def check_base(claim: Claim) -> ClaimVerdict:
            return (
                self._deterministic(claim, allowed_numbers)
                or self._judge(claim, premises=[])
            )

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for verdict in pool.map(check_base, base):
                verdicts[verdict.claim_id] = verdict

        # Walk derived claims in dependency order; propagate premise failures.
        pending = list(derived)
        while pending:
            ready = [c for c in pending if all(d in verdicts or d not in {x.id for x in claims} for d in c.depends_on)]
            if not ready:  # dependency cycle: flag the rest instead of looping
                for claim in pending:
                    verdicts[claim.id] = ClaimVerdict(
                        claim_id=claim.id, status="unsupported", method="chain",
                        rationale="circular dependency between claims",
                    )
                break
            with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                for verdict in pool.map(lambda c: self._check_derived(c, verdicts, allowed_numbers), ready):
                    verdicts[verdict.claim_id] = verdict
            pending = [c for c in pending if c.id not in verdicts]

        return VerificationReport(claims=claims, verdicts=[verdicts[c.id] for c in claims])

    def extract_claims(self, text: str) -> list[Claim]:
        if self.llm is None:
            return heuristic_claims(text)
        try:
            result = self.llm.structured(
                ClaimSet, system=EXTRACT_SYSTEM, prompt=f"<document>\n{text}\n</document>", effort=self.effort
            )
        except Exception as exc:  # degrade, don't fail the workflow
            log.warning("LLM claim extraction failed (%s); using heuristic extraction", exc)
            return heuristic_claims(text)
        known: set[str] = set()
        claims = []
        for claim in result.claims:
            claim.cited_evidence = [e.strip().upper() for e in claim.cited_evidence]
            claim.depends_on = [d for d in claim.depends_on if d in known]
            known.add(claim.id)
            claims.append(claim)
        return claims

    def revise(self, text: str, report: VerificationReport) -> str:
        if not report.flagged:
            return text
        if self.llm is None:
            revised = text
            claims = {c.id: c for c in report.claims}
            for v in report.flagged:
                claim = claims.get(v.claim_id)
                if claim and claim.text in revised:
                    revised = revised.replace(claim.text, "")
            return re.sub(r"\n{3,}", "\n\n", revised).strip()
        prompt = (
            f"<document>\n{text}\n</document>\n\n<fact_check>\n{report.feedback(threshold=0.0)}\n</fact_check>\n\n"
            f"<evidence>\n{self.evidence.render(self._evidence_for_flagged(report))}\n</evidence>"
        )
        return response_text(self.llm.create(messages=[{"role": "user", "content": prompt}], system=REVISE_SYSTEM))

    # -- layers -------------------------------------------------------------
    def _deterministic(self, claim: Claim, allowed_numbers: set[str]) -> ClaimVerdict | None:
        missing = [eid for eid in claim.cited_evidence if not self.evidence.exists(eid)]
        if missing:
            return ClaimVerdict(
                claim_id=claim.id, status="unsupported", method="deterministic",
                rationale=f"cites evidence that does not exist: {', '.join(missing)}",
            )
        if claim.kind in ("numeric", "factual"):
            figures = _normalise_numbers(claim.text)
            if figures:
                cited = self.evidence.get_many(claim.cited_evidence)
                grounded = allowed_numbers.union(
                    *(_normalise_numbers(f"{e.title or ''} {e.content}") for e in cited)
                )
                ungrounded = sorted(figures - grounded)
                if ungrounded:
                    where = ", ".join(claim.cited_evidence) or "no cited evidence"
                    return ClaimVerdict(
                        claim_id=claim.id, status="unsupported", method="deterministic",
                        evidence_ids=claim.cited_evidence,
                        rationale=f"figure(s) {', '.join(ungrounded)} not found in {where} or the brief",
                    )
                if not claim.cited_evidence and figures <= allowed_numbers:
                    return ClaimVerdict(
                        claim_id=claim.id, status="supported", method="deterministic",
                        rationale="restates figures given in the brief",
                    )
        if self.llm is None:
            return self._lexical(claim)
        return None

    def _lexical(self, claim: Claim) -> ClaimVerdict:
        """Offline stand-in for the LLM judge: token overlap with cited evidence."""
        if claim.kind == "recommendation":
            return ClaimVerdict(claim_id=claim.id, status="supported", method="lexical", rationale="planning choice")
        cited = self.evidence.get_many(claim.cited_evidence)
        if not cited:
            return ClaimVerdict(claim_id=claim.id, status="unsupported", method="lexical", rationale="no evidence cited")
        words = _tokens(_CITATION_GROUP.sub("", claim.text))
        support = _tokens(" ".join(f"{e.title or ''} {e.content}" for e in cited))
        overlap = len(words & support) / max(len(words), 1)
        ok = overlap >= self.lexical_threshold
        return ClaimVerdict(
            claim_id=claim.id, status="supported" if ok else "unsupported", method="lexical",
            evidence_ids=[e.id for e in cited],
            rationale=f"{overlap:.0%} term overlap with cited evidence",
        )

    def _check_derived(
        self, claim: Claim, verdicts: dict[str, ClaimVerdict], allowed_numbers: set[str]
    ) -> ClaimVerdict:
        broken = [d for d in claim.depends_on if d in verdicts and verdicts[d].status != "supported"]
        if broken:
            return ClaimVerdict(
                claim_id=claim.id, status="unsupported", method="chain",
                rationale=f"rests on unverified premise(s) {', '.join(broken)}",
            )
        if claim.kind == "recommendation":
            return ClaimVerdict(
                claim_id=claim.id, status="supported", method="chain",
                rationale="planning choice; all premises verified",
            )
        deterministic = self._deterministic(claim, allowed_numbers)
        if deterministic is not None:
            return deterministic
        return self._judge(claim, premises=[d for d in claim.depends_on if d in verdicts])

    def _judge(self, claim: Claim, premises: list[str]) -> ClaimVerdict:
        assert self.llm is not None
        if claim.kind == "recommendation" and not claim.depends_on:
            return ClaimVerdict(claim_id=claim.id, status="supported", method="chain", rationale="planning choice")
        items = self._evidence_for(claim)
        if not items:
            return ClaimVerdict(
                claim_id=claim.id, status="unsupported", method="deterministic",
                rationale="no evidence in the store relates to this claim",
            )
        premise_text = "\n".join(f"- {p}" for p in premises) or "(none)"
        prompt = (
            f"<evidence>\n{self.evidence.render(items)}\n</evidence>\n\n"
            f"<verified_premises>\n{premise_text}\n</verified_premises>\n\n"
            f"<claim kind=\"{claim.kind}\">\n{claim.text}\n</claim>"
        )
        try:
            result = self.llm.structured(LLMVerdict, system=VERIFY_SYSTEM, prompt=prompt, effort=self.effort)
        except Exception as exc:
            log.warning("verification of %s failed (%s); falling back to lexical check", claim.id, exc)
            return self._lexical(claim)
        return ClaimVerdict(
            claim_id=claim.id, status=result.status, method="llm",
            evidence_ids=[e.upper() for e in result.evidence_ids], rationale=result.rationale,
        )

    def _evidence_for(self, claim: Claim) -> list[Evidence]:
        items = {e.id: e for e in self.evidence.get_many(claim.cited_evidence)}
        for item in self.evidence.search(claim.text, k=self.top_k):
            items.setdefault(item.id, item)
        return list(items.values())

    def _evidence_for_flagged(self, report: VerificationReport) -> list[Evidence]:
        claims = {c.id: c for c in report.claims}
        items: dict[str, Evidence] = {}
        for v in report.flagged:
            if v.claim_id in claims:
                for item in self._evidence_for(claims[v.claim_id]):
                    items.setdefault(item.id, item)
        return list(items.values())


@dataclass
class VerificationLedger:
    """Per-task record of every verification attempt in a run."""

    attempts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def record(self, task: str, report: VerificationReport, passed: bool) -> None:
        with self._lock:
            self.attempts.setdefault(task, []).append({**report.summary(), "passed": passed})

    def summary(self) -> dict[str, Any]:
        tasks = {}
        firsts, finals = [], []
        for task, attempts in self.attempts.items():
            first, final = attempts[0]["hallucination_rate"], attempts[-1]["hallucination_rate"]
            firsts.append(first)
            finals.append(final)
            tasks[task] = {"attempts": len(attempts), "first_rate": first, "final_rate": final,
                           "passed": attempts[-1]["passed"]}
        initial = sum(firsts) / len(firsts) if firsts else 0.0
        final = sum(finals) / len(finals) if finals else 0.0
        reduction = (initial - final) / initial if initial else 0.0
        return {
            "tasks": tasks,
            "mean_first_attempt_rate": round(initial, 4),
            "mean_final_rate": round(final, 4),
            "relative_reduction": round(reduction, 4),
        }


class VerificationGuardrail:
    """CrewAI task guardrail running the verification protocol on task output."""

    def __init__(
        self,
        verifier: ChainOfVerification,
        *,
        task_name: str,
        threshold: float,
        ledger: VerificationLedger,
        allowed_facts: str = "",
        on_report: Any = None,
    ):
        self.verifier = verifier
        self.task_name = task_name
        self.threshold = threshold
        self.ledger = ledger
        self.allowed_facts = allowed_facts
        self.on_report = on_report

    # No return annotation: CrewAI validates annotations and rejects string forms.
    def __call__(self, output):
        text = getattr(output, "raw", None) or str(output)
        report = self.verifier.verify(text, allowed_facts=self.allowed_facts)
        passed = report.passed(self.threshold)
        self.ledger.record(self.task_name, report, passed)
        if self.on_report is not None:
            self.on_report(self.task_name, report, passed)
        if passed:
            return True, text
        return False, report.feedback(self.threshold)
