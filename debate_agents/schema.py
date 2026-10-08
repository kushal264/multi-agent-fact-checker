"""
schema.py
=========
Defines the data structures that flow between agents.

LEARNING NOTE:
    In multi-agent systems, the single most important design decision is
    "what does the shared state look like?" Every agent reads from and/or
    writes to this state. If you design this well, adding a new agent later
    (e.g. a second rebuttal round, or a third "moderator" agent) is trivial
    -- you just add a field and a new function that consumes/produces it.

    We use Python @dataclasses here for clarity. In a LangGraph version of
    this project, this exact same shape would become a TypedDict passed as
    the graph's "state" -- the concepts are identical, only the plumbing
    differs. See the README for notes on porting this to LangGraph.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class EvidenceSnippet:
    evidence_id: str
    text: str
    score: float

    # V2 provenance fields
    source_type: str = "local"
    title: str = ""
    url: str = ""
    domain: str = ""
    published_date: str | None = None

    # Raw retrieval score from the retrieval system.
    retrieval_score: float = 0.0

    # Reserved for the future source credibility layer.
    # We will calculate this properly in a later V2 step.
    credibility_score: float = 0.0

@dataclass
class Argument:
    stance: str
    text: str
    cited_evidence_ids: List[str] = field(default_factory=list)


@dataclass
class GroundingReport:
    valid_citation_count: int
    invalid_citation_count: int
    avg_overlap_score: float
    grounding_score: float


@dataclass
class JudgeVerdict:
    verdict: str
    confidence: float
    reasoning: str
    for_score: float
    against_score: float

@dataclass
class DebateResult:
    claim: str
    claim_type: str

    evidence: List[EvidenceSnippet]

    for_argument: Argument
    against_argument: Argument

    for_grounding: GroundingReport
    against_grounding: GroundingReport

    verdict: JudgeVerdict

    for_rebuttal: Argument | None = None
    against_rebuttal: Argument | None = None

    for_rebuttal_grounding: GroundingReport | None = None
    against_rebuttal_grounding: GroundingReport | None = None

    evidence_sufficient: bool = True
    evidence_reason: str = ""

    rejected_evidence: List[EvidenceSnippet] = field(
        default_factory=list
    )

    retrieval_mode: str = "local"