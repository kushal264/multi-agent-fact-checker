"""
claim_classifier.py
===================

Classifies incoming claims before evidence retrieval.

Current categories:

FACTUAL
COMPARATIVE
SUBJECTIVE
"""

import re


FACTUAL_PATTERNS = [
    r"\bis\b",
    r"\bare\b",
    r"\bwas\b",
    r"\bwere\b",
    r"\bhas\b",
    r"\bhave\b",
    r"\bdid\b",
    r"\bdoes\b",
    r"\bdo\b",
    r"\bcauses\b",
    r"\bcaused\b",
    r"\bcontains\b",
    r"\bconsists\b",
    r"\bwon\b",
    r"\bscored\b",
]

COMPARATIVE_PATTERNS = [
    r"\bmore than\b",
    r"\bless than\b",
    r"\bhigher than\b",
    r"\blower than\b",
    r"\bgreater than\b",
    r"\bfaster than\b",
    r"\bslower than\b",
    r"\bolder than\b",
    r"\byounger than\b",
    r"\bbetter than\b",
    r"\bworse than\b",
    r"\bmore\b.*\bthan\b",
    r"\bless\b.*\bthan\b",
]

SUBJECTIVE_PATTERNS = [
    r"\bbest\b",
    r"\bgreatest\b",
    r"\bworst\b",
    r"\bmost beautiful\b",
    r"\bbest ever\b",
    r"\bgreatest ever\b",
    r"\bbetter\b",
    r"\bworth\b",
    r"\bbeautiful\b",
    r"\bamazing\b",
    r"\bexcellent\b",
    r"\boverrated\b",
    r"\bunderrated\b",
]


def classify_claim(claim: str) -> str:
    """
    Return:

    FACTUAL
    COMPARATIVE
    SUBJECTIVE
    """

    text = claim.lower().strip()

    # --------------------------------------------------------
    # Subjective first
    # --------------------------------------------------------

    for pattern in SUBJECTIVE_PATTERNS:
        if re.search(pattern, text):
            return "SUBJECTIVE"

    # --------------------------------------------------------
    # Comparative
    # --------------------------------------------------------

    for pattern in COMPARATIVE_PATTERNS:
        if re.search(pattern, text):
            return "COMPARATIVE"

    # --------------------------------------------------------
    # Default
    # --------------------------------------------------------

    return "FACTUAL"