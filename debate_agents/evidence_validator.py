"""
evidence_validator.py
====================

V2 evidence validation layer.

Responsibilities:

1. Estimate source credibility.
2. Filter clearly weak sources.
3. Remove duplicate evidence.
4. Determine whether enough reliable evidence exists.
5. Return a validated evidence pool.

Important:
retrieval_score != credibility_score
"""

from dataclasses import dataclass
from typing import List
from urllib.parse import urlparse

from .schema import EvidenceSnippet


# ============================================================
# SOURCE QUALITY RULES
# ============================================================

HIGH_CREDIBILITY_DOMAINS = {
    "fifa.com",
    "uefa.com",
    "olympics.com",
    "gov",
    "gov.in",
    "gov.uk",
    "nasa.gov",
    "who.int",
    "nih.gov",
    "cdc.gov",
    "un.org",
    "edu",
}

MEDIUM_CREDIBILITY_DOMAINS = {
    "reuters.com",
    "apnews.com",
    "bbc.com",
    "bbc.co.uk",
    "espn.com",
    "sportsnet.ca",
    "theguardian.com",
    "nytimes.com",
    "washingtonpost.com",
    "nature.com",
    "sciencedirect.com",
    "springer.com",
}

LOW_CREDIBILITY_DOMAINS = {
    "facebook.com",
    "instagram.com",
    "tiktok.com",
    "x.com",
    "twitter.com",
    "reddit.com",
}


def _normalize_domain(domain: str) -> str:
    """
    Normalize a domain so subdomains can be handled more easily.
    """

    domain = domain.lower().strip()

    if domain.startswith("www."):
        domain = domain[4:]

    return domain


def source_credibility_score(
    evidence: EvidenceSnippet,
) -> float:
    """
    Estimate source credibility.

    This is a heuristic, not a claim that a source is objectively
    trustworthy. It is intentionally separate from retrieval score.
    """

    # Local curated corpus.
    if evidence.source_type == "local":
        return 0.85

    domain = _normalize_domain(
        evidence.domain
        or urlparse(evidence.url).netloc
    )

    # Social media / user-generated platforms.
    for blocked in LOW_CREDIBILITY_DOMAINS:
        if domain == blocked or domain.endswith("." + blocked):
            return 0.20

    # Official / institutional sources.
    for trusted in HIGH_CREDIBILITY_DOMAINS:

        if trusted in {"gov", "edu"}:

            if domain.endswith(".gov") or domain.endswith(".edu"):
                return 0.95

        elif domain == trusted or domain.endswith("." + trusted):
            return 0.95

    # Established publishers.
    for trusted in MEDIUM_CREDIBILITY_DOMAINS:
        if domain == trusted or domain.endswith("." + trusted):
            return 0.80

    # Wikipedia.
    if domain == "wikipedia.org" or domain.endswith(
        ".wikipedia.org"
    ):
        return 0.70

    # Unknown web source.
    return 0.45


def validate_evidence_item(
    evidence: EvidenceSnippet,
    min_credibility: float = 0.40,
) -> bool:
    """
    Decide whether one evidence item is acceptable.
    """

    credibility = source_credibility_score(evidence)

    evidence.credibility_score = credibility

    # Reject clearly weak sources.
    if credibility < min_credibility:
        return False

    # Reject empty evidence.
    if not evidence.text.strip():
        return False

    return True


@dataclass
class EvidenceValidationResult:
    accepted: List[EvidenceSnippet]
    rejected: List[EvidenceSnippet]
    sufficient: bool
    reason: str


class EvidenceValidator:
    """
    Validates an evidence pool after retrieval.
    """

    def __init__(
        self,
        min_credibility: float = 0.40,
        min_evidence: int = 2,
        min_relevant_score: float = 0.45,
    ):
        self.min_credibility = min_credibility
        self.min_evidence = min_evidence
        self.min_relevant_score = min_relevant_score

    def validate(
        self,
        evidence: List[EvidenceSnippet],
    ) -> EvidenceValidationResult:

        accepted = []
        rejected = []

        seen = set()

        for item in evidence:

            # -----------------------------------------------
            # Duplicate check
            # -----------------------------------------------

            identifier = (
                item.url
                if item.url
                else f"{item.source_type}:{item.text[:200]}"
            )

            if identifier in seen:
                continue

            seen.add(identifier)

            # -----------------------------------------------
            # Source credibility
            # -----------------------------------------------

            is_valid = validate_evidence_item(
                item,
                min_credibility=self.min_credibility,
            )

            if not is_valid:
                rejected.append(item)
                continue

            # -----------------------------------------------
            # Retrieval relevance
            # -----------------------------------------------

            relevance = (
                item.retrieval_score
                if item.retrieval_score
                else item.score
            )

            if relevance < self.min_relevant_score:
                rejected.append(item)
                continue

            accepted.append(item)

        # Highest-quality evidence first.
        accepted.sort(
            key=lambda item: (
                0.60
                * (
                    item.retrieval_score
                    if item.retrieval_score
                    else item.score
                )
                + 0.40 * item.credibility_score
            ),
            reverse=True,
        )

        # ----------------------------------------------------
        # SUFFICIENCY
        # ----------------------------------------------------

        strong_evidence = [
            item
            for item in accepted
            if (
                (
                    item.retrieval_score
                    if item.retrieval_score
                    else item.score
                )
                >= self.min_relevant_score
                and item.credibility_score >= 0.50
            )
        ]

        sufficient = len(strong_evidence) >= self.min_evidence

        if sufficient:
            reason = (
                f"{len(strong_evidence)} sufficiently relevant "
                f"and credible evidence items found."
            )
        else:
            reason = (
                "Insufficient reliable evidence was found "
                "to evaluate the claim confidently."
            )

        return EvidenceValidationResult(
            accepted=accepted,
            rejected=rejected,
            sufficient=sufficient,
            reason=reason,
        )