from types import SimpleNamespace

from marketing_crew.verification.cove import (
    FEEDBACK_MARKER,
    ChainOfVerification,
    Claim,
    ClaimSet,
    LLMVerdict,
    VerificationGuardrail,
    VerificationLedger,
    cited_ids,
    heuristic_claims,
)


def seed(evidence):
    places = evidence.add(source="find_nearby_businesses", content='[{"name": "Fogline Coffee Bar", "distance_m": 97}]')
    wiki = evidence.add(source="wikipedia_summary", content="Cold brew is steeped 12 to 24 hours; smoother and less acidic.")
    return places.id, wiki.id


def test_citation_parsing_and_heuristic_kinds():
    assert cited_ids("x [E1, E3] y [E3]") == ["E1", "E3"]
    kinds = {c.text[:6]: c.kind for c in heuristic_claims(
        "Fogline Coffee Bar is 97 m away [E1].\nWe should allocate $5,000 to events.\nCold brew is smooth [E2].")}
    assert kinds == {"Foglin": "numeric", "We sho": "recommendation", "Cold b": "factual"}


def test_deterministic_layer_flags_fabrications(evidence):
    places, wiki = seed(evidence)
    text = (
        f"Fogline Coffee Bar is 97 m from the centre [{places}].\n"
        f"Cold brew is steeped 12 to 24 hours, smoother and less acidic [{wiki}].\n"
        "Surveys show 87% of residents buy cold brew weekly [E99].\n"
        f"Fogline Coffee Bar is 450 m away [{places}].\n"
    )
    report = ChainOfVerification(evidence).verify(text)
    flagged = {v.claim_id: v.rationale for v in report.flagged}
    assert set(flagged) == {"C3", "C4"}
    assert "E99" in flagged["C3"] and "450" in flagged["C4"]
    assert report.hallucination_rate == 0.5
    assert FEEDBACK_MARKER in report.feedback(0.1)


def test_brief_facts_ground_figures(evidence):
    report = ChainOfVerification(evidence).verify("The plan runs for 90 days on a 25,000 budget.", allowed_facts="horizon 90 budget 25000")
    assert report.flagged == []


def test_offline_revise_drops_flagged_claims(evidence):
    places, _ = seed(evidence)
    text = f"Fogline Coffee Bar is 97 m away [{places}].\nSurveys show 87% love it [E99]."
    cove = ChainOfVerification(evidence)
    revised = cove.revise(text, cove.verify(text))
    assert "87%" not in revised and "Fogline" in revised


class FakeLLM:
    """Stands in for ClaudeClient.structured with canned claim sets / verdicts."""

    def __init__(self, claims, verdicts):
        self.claims, self.verdicts, self.prompts = claims, verdicts, []

    def structured(self, schema, *, prompt, system=None, effort=None, **_):
        self.prompts.append(prompt)
        if schema is ClaimSet:
            return ClaimSet(claims=self.claims)
        claim_text = prompt.split("<claim", 1)[1]
        return next(v for key, v in self.verdicts.items() if key in claim_text)


def test_chain_propagates_premise_failures_and_judges_in_isolation(evidence):
    places, wiki = seed(evidence)
    claims = [
        Claim(id="C1", text="Fogline is the nearest cafe", kind="factual", cited_evidence=[places], depends_on=[]),
        Claim(id="C2", text="Demand for cold brew is booming locally", kind="factual", cited_evidence=[], depends_on=[]),
        Claim(id="C3", text="So a cold-brew bar faces little competition", kind="inference", cited_evidence=[], depends_on=["C1", "C2"]),
        Claim(id="C4", text="Open a second location next year", kind="recommendation", cited_evidence=[], depends_on=["C3"]),
    ]
    llm = FakeLLM(claims, {
        "Fogline": LLMVerdict(status="supported", evidence_ids=[places], rationale="listed"),
        "booming": LLMVerdict(status="unsupported", evidence_ids=[], rationale="no demand data"),
    })
    report = ChainOfVerification(evidence, llm).verify("draft text that the judge must never see")
    status = {v.claim_id: (v.status, v.method) for v in report.verdicts}
    assert status == {"C1": ("supported", "llm"), "C2": ("unsupported", "llm"),
                      "C3": ("unsupported", "chain"), "C4": ("unsupported", "chain")}
    assert report.hallucination_rate == 2 / 3
    judge_prompts = llm.prompts[1:]
    assert judge_prompts and all("draft text" not in p for p in judge_prompts)  # factored verification


def test_guardrail_feedback_and_ledger(evidence):
    places, _ = seed(evidence)
    ledger = VerificationLedger()
    guard = VerificationGuardrail(ChainOfVerification(evidence), task_name="t", threshold=0.1, ledger=ledger)
    ok, feedback = guard(SimpleNamespace(raw="Surveys show 87% love it [E99]."))
    assert not ok and FEEDBACK_MARKER in feedback
    ok, text = guard(SimpleNamespace(raw=f"Fogline Coffee Bar is 97 m away [{places}]."))
    assert ok and "Fogline" in text
    summary = ledger.summary()
    assert summary["tasks"]["t"] == {"attempts": 2, "first_rate": 1.0, "final_rate": 0.0, "passed": True}
    assert summary["relative_reduction"] == 1.0
