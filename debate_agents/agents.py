from typing import List

from .schema import (
    Argument,
    EvidenceSnippet,
    GroundingReport,
    JudgeVerdict,
)

from .rubric import (
    check_grounding,
    stance_score,
    composite_score,
)


def _format_evidence(evidence: List[EvidenceSnippet]) -> str:
    return "\n".join(
        f"[{item.evidence_id}] {item.text}"
        for item in evidence
    )


def _extract_citations(text: str) -> List[str]:
    """
    Extract citations written like [E1], [E2], etc.
    """
    citations = []

    for token in text.split():
        cleaned = token.strip(".,;:()[]")

        if cleaned.startswith("E") and cleaned[1:].isdigit():
            citations.append(cleaned)

    return list(dict.fromkeys(citations))


class AdvocateAgent:
    """
    Produces either a FOR or AGAINST argument.
    """

    def __init__(self, llm, stance: str):
        self.llm = llm
        self.stance = stance

    def argue(
        self,
        claim: str,
        evidence: List[EvidenceSnippet],
        opponent_argument: str | None = None,
    ) -> Argument:

        evidence_text = _format_evidence(evidence)

        if opponent_argument:
            task = f"""
You are participating in an adversarial fact-checking debate.

CLAIM:
{claim}

YOUR STANCE:
{self.stance}

AVAILABLE EVIDENCE:
{evidence_text}

OPPONENT'S ARGUMENT:
{opponent_argument}

Your task is to REBUT the opponent.

Rules:
1. Attack the opponent's reasoning directly.
2. Use only the provided evidence.
3. Cite evidence using [E1], [E2], etc.
4. Do not invent evidence.
5. Explain why the evidence supports your position.
6. Keep the response concise and analytical.
7. Explicitly identify the opponent's weakest point.
8. Do not simply repeat your opening argument.

Return only the rebuttal.
"""
        else:
            task = f"""
You are participating in an adversarial fact-checking debate.

CLAIM:
{claim}

YOUR STANCE:
{self.stance}

AVAILABLE EVIDENCE:
{evidence_text}

Build the strongest possible opening argument for your assigned stance.

Rules:
1. Use only the provided evidence.
2. Cite factual claims using [E1], [E2], etc.
3. Never invent evidence or citations.
4. Explain how the evidence supports your position.
5. Directly address the claim.
6. Keep the argument concise and analytical.
7. Do not mention information that is not present in the evidence.

Return only the argument.
"""

        response_text = self.llm.generate(task).strip()

        citations = _extract_citations(response_text)

        return Argument(
            stance=self.stance,
            text=response_text,
            cited_evidence_ids=citations,
        )


class JudgeAgent:
    """
    Deterministic judge.

    Gemini is used for advocacy/rebuttal.
    The final judge uses evidence grounding + stance scoring
    rather than making another expensive Gemini request.
    """

    def _quality_score(self, argument: Argument) -> float:
        score = 0.5

        # Rewards citations.
        if argument.cited_evidence_ids:
            score += 0.2

        # Reasonable argument length.
        word_count = len(argument.text.split())

        if word_count >= 20:
            score += 0.1

        if word_count <= 150:
            score += 0.1

        # Some degree of structure/content.
        if any(
            keyword in argument.text.lower()
            for keyword in [
                "because",
                "therefore",
                "however",
                "evidence",
                "supports",
                "contradicts",
            ]
        ):
            score += 0.1

        return min(score, 1.0)

    def judge(
        self,
        claim: str,
        evidence: List[EvidenceSnippet],
        for_argument: Argument,
        against_argument: Argument,
        for_rebuttal: Argument | None = None,
        against_rebuttal: Argument | None = None,
    ):

        # ---------------------------------------------------------
        # 1. Grounding of opening arguments
        # ---------------------------------------------------------

        for_grounding = check_grounding(
            for_argument,
            evidence,
        )

        against_grounding = check_grounding(
            against_argument,
            evidence,
        )

        # ---------------------------------------------------------
        # 2. Grounding of rebuttals
        # ---------------------------------------------------------

        for_rebuttal_grounding = None
        against_rebuttal_grounding = None

        if for_rebuttal:
            for_rebuttal_grounding = check_grounding(
                for_rebuttal,
                evidence,
            )

        if against_rebuttal:
            against_rebuttal_grounding = check_grounding(
                against_rebuttal,
                evidence,
            )

        # ---------------------------------------------------------
        # 3. Stance scores
        # ---------------------------------------------------------

        for_stance = stance_score(
            claim,
            for_argument,
            evidence,
        )

        against_stance = stance_score(
            claim,
            against_argument,
            evidence,
        )

        # ---------------------------------------------------------
        # 4. Opening quality
        # ---------------------------------------------------------

        for_quality = self._quality_score(for_argument)
        against_quality = self._quality_score(against_argument)

        for_score = composite_score(
            for_grounding.grounding_score,
            for_quality,
            for_stance,
        )

        against_score = composite_score(
            against_grounding.grounding_score,
            against_quality,
            against_stance,
        )

        # ---------------------------------------------------------
        # 5. Rebuttal adjustment
        # ---------------------------------------------------------

        if for_rebuttal and for_rebuttal_grounding:
            rebuttal_quality = self._quality_score(for_rebuttal)

            for_score = (
                0.85 * for_score
                + 0.15
                * (
                    0.6 * for_rebuttal_grounding.grounding_score
                    + 0.4 * rebuttal_quality
                )
            )

        if against_rebuttal and against_rebuttal_grounding:
            rebuttal_quality = self._quality_score(against_rebuttal)

            against_score = (
                0.85 * against_score
                + 0.15
                * (
                    0.6 * against_rebuttal_grounding.grounding_score
                    + 0.4 * rebuttal_quality
                )
            )

        # ---------------------------------------------------------
        # 6. Final verdict
        # ---------------------------------------------------------

        margin = abs(for_score - against_score)

        if margin < 0.10:
            verdict_text = "UNCERTAIN"
        elif for_score > against_score:
            verdict_text = "TRUE"
        else:
            verdict_text = "FALSE"

        confidence = min(1.0, margin * 2)

        reasoning = (
            f"FOR score={for_score:.3f}, "
            f"AGAINST score={against_score:.3f}. "
            f"Score margin={margin:.3f}. "
            f"The verdict is based on evidence grounding, "
            f"stance consistency, argument quality, and rebuttal quality."
        )

        verdict = JudgeVerdict(
            verdict=verdict_text,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            for_score=round(for_score, 3),
            against_score=round(against_score, 3),
        )

        return (
            verdict,
            for_grounding,
            against_grounding,
            for_rebuttal_grounding,
            against_rebuttal_grounding,
        )