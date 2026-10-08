"""
graph.py
========

LangGraph orchestration for the V2 hybrid/open-world fact checker.

Flow:

START
  ↓
classify_claim
  ↓
 ┌──────────────────────────────────────┐
 │                 │                    │
FACTUAL       COMPARATIVE           SUBJECTIVE
 │                 │                    │
 └──────────┬──────┘                    ↓
            ↓                       SUBJECTIVE
      retrieve_evidence                 ↓
            ↓                          END
      validate_evidence
            ↓
 ┌───────────────────────────────┐
 │                               │
sufficient                  insufficient
 │                               │
 ▼                               ▼
start_debate                 UNCERTAIN
 │
 ├── FOR opening
 │
 └── AGAINST opening
        ↓
     rebuttals
        ↓
       judge
        ↓
       END
"""

from typing import TypedDict, Any

from langgraph.graph import StateGraph, START, END

from .claim_classifier import classify_claim

from .agents import AdvocateAgent, JudgeAgent

from .schema import (
    Argument,
    DebateResult,
    EvidenceSnippet,
    GroundingReport,
    JudgeVerdict,
)

from .evidence_validator import EvidenceValidator


# ============================================================
# LANGGRAPH STATE
# ============================================================

class DebateState(TypedDict, total=False):

    # --------------------------------------------------------
    # Claim
    # --------------------------------------------------------

    claim: str

    # FACTUAL / COMPARATIVE / SUBJECTIVE
    claim_type: str

    # --------------------------------------------------------
    # Retrieval
    # --------------------------------------------------------

    # Raw evidence returned by HybridRetriever
    raw_evidence: list[EvidenceSnippet]

    # Evidence accepted by EvidenceValidator
    evidence: list[EvidenceSnippet]

    # Evidence rejected by EvidenceValidator
    rejected_evidence: list[EvidenceSnippet]

    # local / web / hybrid / none
    retrieval_mode: str

    # --------------------------------------------------------
    # Evidence validation
    # --------------------------------------------------------

    evidence_sufficient: bool
    evidence_reason: str

    # --------------------------------------------------------
    # Debate
    # --------------------------------------------------------

    for_argument: Argument
    against_argument: Argument

    for_rebuttal: Argument
    against_rebuttal: Argument

    # --------------------------------------------------------
    # Grounding
    # --------------------------------------------------------

    for_grounding: GroundingReport
    against_grounding: GroundingReport

    for_rebuttal_grounding: GroundingReport
    against_rebuttal_grounding: GroundingReport

    # --------------------------------------------------------
    # Final result
    # --------------------------------------------------------

    verdict: Any
    result: DebateResult


# ============================================================
# BUILD GRAPH
# ============================================================

def build_debate_graph(
    retriever,
    llm,
    k_evidence: int = 6,
    rebuttal_round: bool = True,
    validator: EvidenceValidator | None = None,
):

    # ========================================================
    # VALIDATOR
    # ========================================================

    if validator is None:
        validator = EvidenceValidator()

    # ========================================================
    # AGENTS
    # ========================================================

    for_agent = AdvocateAgent(
        llm,
        stance="FOR",
    )

    against_agent = AdvocateAgent(
        llm,
        stance="AGAINST",
    )

    judge = JudgeAgent()

    # ========================================================
    # NODE 1 — CLAIM CLASSIFICATION
    # ========================================================

    def classify_claim_node(
        state: DebateState,
    ):
        claim_type = classify_claim(
            state["claim"]
        )

        return {
            "claim_type": claim_type,
        }

    # ========================================================
    # CLAIM ROUTER
    # ========================================================

    def route_after_classification(
        state: DebateState,
    ):
        claim_type = state.get(
            "claim_type",
            "FACTUAL",
        )

        if claim_type == "SUBJECTIVE":
            return "subjective_claim"

        return "retrieve_evidence"

    # ========================================================
    # NODE 2 — SUBJECTIVE CLAIM
    # ========================================================

    def subjective_claim(
        state: DebateState,
    ):

        verdict = JudgeVerdict(
            verdict="SUBJECTIVE",

            confidence=1.0,

            reasoning=(
                "The claim contains an evaluative or "
                "opinion-based judgment that cannot be "
                "reduced to a simple factual TRUE/FALSE "
                "statement without defining objective "
                "evaluation criteria."
            ),

            for_score=0.0,
            against_score=0.0,
        )

        # ----------------------------------------------------
        # Empty grounding
        # ----------------------------------------------------

        empty_grounding = GroundingReport(
            valid_citation_count=0,
            invalid_citation_count=0,
            avg_overlap_score=0.0,
            grounding_score=0.0,
        )

        # ----------------------------------------------------
        # Placeholder arguments
        # ----------------------------------------------------

        for_argument = Argument(
            stance="FOR",

            text=(
                "Debate not executed because the claim "
                "was classified as subjective."
            ),

            cited_evidence_ids=[],
        )

        against_argument = Argument(
            stance="AGAINST",

            text=(
                "Debate not executed because the claim "
                "was classified as subjective."
            ),

            cited_evidence_ids=[],
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        result = DebateResult(
            claim=state["claim"],

            claim_type="SUBJECTIVE",

            evidence=[],

            for_argument=for_argument,
            against_argument=against_argument,

            for_grounding=empty_grounding,
            against_grounding=empty_grounding,

            verdict=verdict,

            for_rebuttal=None,
            against_rebuttal=None,

            for_rebuttal_grounding=None,
            against_rebuttal_grounding=None,

            evidence_sufficient=False,

            evidence_reason=(
                "Subjective claim: objective evaluation "
                "criteria must be defined first."
            ),

            rejected_evidence=[],

            retrieval_mode="none",
        )

        return {
            "claim_type": "SUBJECTIVE",
            "verdict": verdict,
            "result": result,
            "evidence_sufficient": False,
        }

    # ========================================================
    # NODE 3 — RETRIEVE EVIDENCE
    # ========================================================

    def retrieve_evidence(
        state: DebateState,
    ):

        raw_evidence = retriever.retrieve(
            state["claim"],
            k=k_evidence,
        )

        # Determine retrieval mode automatically.
        source_types = {
            item.source_type
            for item in raw_evidence
        }

        if (
            "web" in source_types
            and "local" in source_types
        ):
            retrieval_mode = "hybrid"

        elif "web" in source_types:
            retrieval_mode = "web"

        elif "local" in source_types:
            retrieval_mode = "local"

        else:
            retrieval_mode = "none"

        return {
            "raw_evidence": raw_evidence,
            "retrieval_mode": retrieval_mode,
        }

    # ========================================================
    # NODE 4 — VALIDATE EVIDENCE
    # ========================================================

    def validate_evidence(
        state: DebateState,
    ):

        validation = validator.validate(
            state["raw_evidence"]
        )

        return {
            # Only accepted evidence reaches agents.
            "evidence": validation.accepted,

            # Keep rejected evidence for transparency.
            "rejected_evidence": validation.rejected,

            # Sufficiency decision.
            "evidence_sufficient": validation.sufficient,

            # Human-readable explanation.
            "evidence_reason": validation.reason,
        }

    # ========================================================
    # EVIDENCE ROUTER
    # ========================================================

    def route_after_validation(
        state: DebateState,
    ):

        if state.get(
            "evidence_sufficient",
            False,
        ):
            return "start_debate"

        return "insufficient_evidence"

    # ========================================================
    # NODE 5 — START DEBATE
    # ========================================================

    def start_debate(
        state: DebateState,
    ):
        """
        Explicit branching point before
        FOR and AGAINST agents.
        """

        return {}

    # ========================================================
    # NODE 6 — FOR OPENING
    # ========================================================

    def for_opening(
        state: DebateState,
    ):

        argument = for_agent.argue(
            state["claim"],
            state["evidence"],
        )

        return {
            "for_argument": argument,
        }

    # ========================================================
    # NODE 7 — AGAINST OPENING
    # ========================================================

    def against_opening(
        state: DebateState,
    ):

        argument = against_agent.argue(
            state["claim"],
            state["evidence"],
        )

        return {
            "against_argument": argument,
        }

    # ========================================================
    # NODE 8 — FOR REBUTTAL
    # ========================================================

    def for_rebuttal(
        state: DebateState,
    ):

        if not rebuttal_round:
            return {}

        rebuttal = for_agent.argue(
            state["claim"],
            state["evidence"],
            opponent_argument=(
                state["against_argument"].text
            ),
        )

        return {
            "for_rebuttal": rebuttal,
        }

    # ========================================================
    # NODE 9 — AGAINST REBUTTAL
    # ========================================================

    def against_rebuttal(
        state: DebateState,
    ):

        if not rebuttal_round:
            return {}

        rebuttal = against_agent.argue(
            state["claim"],
            state["evidence"],
            opponent_argument=(
                state["for_argument"].text
            ),
        )

        return {
            "against_rebuttal": rebuttal,
        }

    # ========================================================
    # NODE 10 — JUDGE
    # ========================================================

    def judge_debate(
        state: DebateState,
    ):

        (
            verdict,
            for_grounding,
            against_grounding,
            for_rebuttal_grounding,
            against_rebuttal_grounding,
        ) = judge.judge(
            claim=state["claim"],
            evidence=state["evidence"],

            for_argument=state["for_argument"],
            against_argument=state["against_argument"],

            for_rebuttal=state.get(
                "for_rebuttal"
            ),

            against_rebuttal=state.get(
                "against_rebuttal"
            ),
        )

        # ----------------------------------------------------
        # Build final DebateResult
        # ----------------------------------------------------

        result = DebateResult(
            claim=state["claim"],

            claim_type=state.get(
                "claim_type",
                "FACTUAL",
            ),

            evidence=state["evidence"],

            # Opening arguments
            for_argument=state["for_argument"],
            against_argument=state["against_argument"],

            # Opening grounding
            for_grounding=for_grounding,
            against_grounding=against_grounding,

            # Final verdict
            verdict=verdict,

            # Rebuttals
            for_rebuttal=state.get(
                "for_rebuttal"
            ),

            against_rebuttal=state.get(
                "against_rebuttal"
            ),

            # Rebuttal grounding
            for_rebuttal_grounding=(
                for_rebuttal_grounding
            ),

            against_rebuttal_grounding=(
                against_rebuttal_grounding
            ),

            # V2 evidence metadata
            evidence_sufficient=True,

            evidence_reason=state.get(
                "evidence_reason",
                "",
            ),

            rejected_evidence=state.get(
                "rejected_evidence",
                [],
            ),

            retrieval_mode=state.get(
                "retrieval_mode",
                "local",
            ),
        )

        return {
            "verdict": verdict,

            "for_grounding": for_grounding,
            "against_grounding": against_grounding,

            "for_rebuttal_grounding": (
                for_rebuttal_grounding
            ),

            "against_rebuttal_grounding": (
                against_rebuttal_grounding
            ),

            "result": result,
        }

    # ========================================================
    # NODE 11 — INSUFFICIENT EVIDENCE
    # ========================================================

    def insufficient_evidence(
        state: DebateState,
    ):

        # ----------------------------------------------------
        # UNCERTAIN verdict
        # ----------------------------------------------------

        verdict = JudgeVerdict(
            verdict="UNCERTAIN",

            confidence=0.0,

            reasoning=state.get(
                "evidence_reason",
                "Insufficient reliable evidence was found.",
            ),

            for_score=0.0,
            against_score=0.0,
        )

        # ----------------------------------------------------
        # Placeholder arguments
        # ----------------------------------------------------

        for_argument = Argument(
            stance="FOR",

            text=(
                "Debate not executed because "
                "reliable evidence was insufficient."
            ),

            cited_evidence_ids=[],
        )

        against_argument = Argument(
            stance="AGAINST",

            text=(
                "Debate not executed because "
                "reliable evidence was insufficient."
            ),

            cited_evidence_ids=[],
        )

        # ----------------------------------------------------
        # Empty grounding
        # ----------------------------------------------------

        empty_grounding = GroundingReport(
            valid_citation_count=0,
            invalid_citation_count=0,
            avg_overlap_score=0.0,
            grounding_score=0.0,
        )

        # ----------------------------------------------------
        # Final result
        # ----------------------------------------------------

        result = DebateResult(
            claim=state["claim"],

            claim_type=state.get(
                "claim_type",
                "FACTUAL",
            ),

            evidence=state.get(
                "evidence",
                [],
            ),

            for_argument=for_argument,
            against_argument=against_argument,

            for_grounding=empty_grounding,
            against_grounding=empty_grounding,

            verdict=verdict,

            for_rebuttal=None,
            against_rebuttal=None,

            for_rebuttal_grounding=None,
            against_rebuttal_grounding=None,

            # V2 metadata
            evidence_sufficient=False,

            evidence_reason=state.get(
                "evidence_reason",
                "Insufficient reliable evidence was found.",
            ),

            rejected_evidence=state.get(
                "rejected_evidence",
                [],
            ),

            retrieval_mode=state.get(
                "retrieval_mode",
                "hybrid",
            ),
        )

        return {
            "verdict": verdict,
            "result": result,
        }

    # ========================================================
    # BUILD STATE GRAPH
    # ========================================================

    builder = StateGraph(
        DebateState
    )

    # ========================================================
    # REGISTER NODES
    # ========================================================

    builder.add_node(
        "classify_claim",
        classify_claim_node,
    )

    builder.add_node(
        "subjective_claim",
        subjective_claim,
    )

    builder.add_node(
        "retrieve_evidence",
        retrieve_evidence,
    )

    builder.add_node(
        "validate_evidence",
        validate_evidence,
    )

    builder.add_node(
        "start_debate",
        start_debate,
    )

    builder.add_node(
        "for_opening",
        for_opening,
    )

    builder.add_node(
        "against_opening",
        against_opening,
    )

    builder.add_node(
        "for_rebuttal",
        for_rebuttal,
    )

    builder.add_node(
        "against_rebuttal",
        against_rebuttal,
    )

    builder.add_node(
        "judge",
        judge_debate,
    )

    builder.add_node(
        "insufficient_evidence",
        insufficient_evidence,
    )

    # ========================================================
    # EDGES
    # ========================================================

    # --------------------------------------------------------
    # START → CLAIM CLASSIFIER
    # --------------------------------------------------------

    builder.add_edge(
        START,
        "classify_claim",
    )

    # --------------------------------------------------------
    # CLAIM CLASSIFIER → NEXT STEP
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "classify_claim",
        route_after_classification,
        {
            "retrieve_evidence":
                "retrieve_evidence",

            "subjective_claim":
                "subjective_claim",
        },
    )

    # Subjective claims terminate here.
    builder.add_edge(
        "subjective_claim",
        END,
    )

    # --------------------------------------------------------
    # RETRIEVAL → VALIDATION
    # --------------------------------------------------------

    builder.add_edge(
        "retrieve_evidence",
        "validate_evidence",
    )

    # --------------------------------------------------------
    # VALIDATION → DEBATE / UNCERTAIN
    # --------------------------------------------------------

    builder.add_conditional_edges(
        "validate_evidence",
        route_after_validation,
        {
            "start_debate":
                "start_debate",

            "insufficient_evidence":
                "insufficient_evidence",
        },
    )

    # --------------------------------------------------------
    # DEBATE FAN-OUT
    # --------------------------------------------------------

    builder.add_edge(
        "start_debate",
        "for_opening",
    )

    builder.add_edge(
        "start_debate",
        "against_opening",
    )

    # --------------------------------------------------------
    # BOTH OPENINGS → REBUTTALS
    # --------------------------------------------------------

    builder.add_edge(
        [
            "for_opening",
            "against_opening",
        ],
        "for_rebuttal",
    )

    builder.add_edge(
        [
            "for_opening",
            "against_opening",
        ],
        "against_rebuttal",
    )

    # --------------------------------------------------------
    # BOTH REBUTTALS → JUDGE
    # --------------------------------------------------------

    builder.add_edge(
        [
            "for_rebuttal",
            "against_rebuttal",
        ],
        "judge",
    )

    # --------------------------------------------------------
    # END
    # --------------------------------------------------------

    builder.add_edge(
        "judge",
        END,
    )

    builder.add_edge(
        "insufficient_evidence",
        END,
    )

    # ========================================================
    # COMPILE
    # ========================================================

    return builder.compile()