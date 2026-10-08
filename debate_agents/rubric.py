"""
rubric.py
=========
Programmatic scoring for evidence grounding and evidence stance.

The system evaluates:
1. Does the argument cite real evidence?
2. Does the argument overlap with the cited evidence?
3. Does the cited evidence SUPPORT or CONTRADICT the claim?

This keeps the final verdict deterministic and auditable.
"""

import re
from typing import List

from .schema import Argument, EvidenceSnippet, GroundingReport


def _words(text: str):
    return set(re.findall(r"[a-z]{3,}", text.lower()))


def _word_overlap(a: str, b: str) -> float:
    """Return lexical overlap between two pieces of text."""
    wa = _words(a)
    wb = _words(b)

    if not wa:
        return 0.0

    return len(wa & wb) / len(wa)


# Words/phrases that frequently indicate that evidence contradicts
# the proposition being discussed.
CONTRADICTION_PATTERNS = [
    r"\bnot\b",
    r"\bnever\b",
    r"\bno\b",
    r"\bmyth\b",
    r"\bdebunked\b",
    r"\bfalse\b",
    r"\bcontradict",
    r"\bcontrary\b",
    r"\bdisprove",
    r"\bdisproves\b",
    r"\brefute",
    r"\brefutes\b",
    r"\bno causal link\b",
    r"\bdoes not\b",
    r"\bdo not\b",
    r"\bdoesn't\b",
    r"\bdon't\b",
    r"\binstead\b",
    r"\bdifferent from\b",
    r"\bonly\b",
    r"\bat least\b",
]


def _contradiction_signal(text: str) -> float:
    """
    Estimate whether evidence contains language indicating contradiction.

    This is intentionally transparent rather than pretending to be
    semantic understanding.
    """
    text = text.lower()

    matches = 0

    for pattern in CONTRADICTION_PATTERNS:
        if re.search(pattern, text):
            matches += 1

    return min(1.0, matches * 0.25)


def evidence_stance(
    claim: str,
    evidence_text: str,
) -> str:
    """
    Classify evidence as SUPPORT, CONTRADICT, or NEUTRAL.

    This is a deterministic heuristic used by the offline backend.
    """

    claim_words = _words(claim)
    evidence_words = _words(evidence_text)

    if not claim_words or not evidence_words:
        return "NEUTRAL"

    overlap = len(claim_words & evidence_words) / len(claim_words)

    contradiction = _contradiction_signal(evidence_text)

    # Strong contradiction signal + reasonable topic overlap.
    if contradiction >= 0.25 and overlap >= 0.15:
        return "CONTRADICT"

    # Evidence discusses the same subject without obvious
    # contradiction language.
    if overlap >= 0.25:
        return "SUPPORT"

    return "NEUTRAL"


def check_grounding(
    argument: Argument,
    evidence: List[EvidenceSnippet],
) -> GroundingReport:
    """
    Programmatically audits an argument's citations.

    Checks:
    - citation exists
    - argument overlaps with cited evidence
    - hallucinated citations are penalized
    """

    valid_ids = {e.evidence_id for e in evidence}
    id_to_text = {e.evidence_id: e.text for e in evidence}

    valid = 0
    invalid = 0
    overlaps = []

    for citation_id in argument.cited_evidence_ids:

        if citation_id in valid_ids:
            valid += 1

            overlaps.append(
                _word_overlap(
                    argument.text,
                    id_to_text[citation_id],
                )
            )

        else:
            invalid += 1

    avg_overlap = (
        sum(overlaps) / len(overlaps)
        if overlaps
        else 0.0
    )

    if not argument.cited_evidence_ids:
        grounding_score = 0.0

    else:
        hallucination_penalty = (
            invalid / len(argument.cited_evidence_ids)
        )

        grounding_score = max(
            0.0,
            avg_overlap - hallucination_penalty,
        )

    return GroundingReport(
        valid_citation_count=valid,
        invalid_citation_count=invalid,
        avg_overlap_score=round(avg_overlap, 3),
        grounding_score=round(grounding_score, 3),
    )


def stance_score(
    claim: str,
    argument: Argument,
    evidence: List[EvidenceSnippet],
) -> float:
    """
    Score whether the cited evidence actually matches the agent's stance.

    FOR:
        SUPPORT evidence -> high
        CONTRADICT evidence -> low

    AGAINST:
        CONTRADICT evidence -> high
        SUPPORT evidence -> low
    """

    id_to_text = {
        e.evidence_id: e.text
        for e in evidence
    }

    if not argument.cited_evidence_ids:
        return 0.0

    scores = []

    for citation_id in argument.cited_evidence_ids:

        if citation_id not in id_to_text:
            scores.append(0.0)
            continue

        stance = evidence_stance(
            claim,
            id_to_text[citation_id],
        )

        if argument.stance == "FOR":

            if stance == "SUPPORT":
                scores.append(1.0)

            elif stance == "CONTRADICT":
                scores.append(0.0)

            else:
                scores.append(0.35)

        else:  # AGAINST

            if stance == "CONTRADICT":
                scores.append(1.0)

            elif stance == "SUPPORT":
                scores.append(0.0)

            else:
                scores.append(0.35)

    return round(sum(scores) / len(scores), 3)


def composite_score(
    grounding_score: float,
    llm_quality_score: float,
    stance_score_value: float = 0.0,
) -> float:
    """
    Final side score.

    Grounding:
        Did the agent cite real evidence?

    Stance:
        Does that evidence actually support the side?

    Quality:
        Is the argument logically consistent?

    Evidence stance receives the largest weight because it is the
    most important signal for factual verification.
    """

    GROUNDING_WEIGHT = 0.25
    STANCE_WEIGHT = 0.55
    QUALITY_WEIGHT = 0.20

    return round(
        GROUNDING_WEIGHT * grounding_score
        + STANCE_WEIGHT * stance_score_value
        + QUALITY_WEIGHT * llm_quality_score,
        3,
    )