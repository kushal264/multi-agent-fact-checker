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
    evidence: List[EvidenceSnippet]

    # Opening arguments
    for_argument: Argument
    against_argument: Argument

    # Grounding for opening arguments
    for_grounding: GroundingReport
    against_grounding: GroundingReport

    # Final verdict
    verdict: JudgeVerdict

    # Rebuttal arguments
    for_rebuttal: Argument | None = None
    against_rebuttal: Argument | None = None

    # Grounding for rebuttals
    for_rebuttal_grounding: GroundingReport | None = None
    against_rebuttal_grounding: GroundingReport | None = None