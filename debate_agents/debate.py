"""
debate.py
=========
Orchestrates the full pipeline. This is the "graph" -- in a LangGraph
version of this project this file's logic would become graph nodes/edges,
but the SEQUENCE of steps is identical either way:

    retrieve -> advocate(FOR) -> advocate(AGAINST) -> [optional rebuttal] -> judge

LEARNING NOTE: notice this function has no LLM-specific code and no
retrieval-specific code in it -- it just wires together the interfaces
defined in llm.py / retriever.py / agents.py. This separation is what
makes multi-agent systems maintainable: you can change the LLM provider,
the retriever, or even add a 3rd advocate, by touching ONE file each,
without ever touching this orchestration logic.
"""
from .agents import AdvocateAgent, JudgeAgent
from .schema import DebateResult


def run_debate(
    claim: str,
    retriever,
    llm,
    k_evidence: int = 4,
    rebuttal_round: bool = True,
) -> DebateResult:

    # =========================================================
    # 1. RETRIEVE EVIDENCE
    # =========================================================

    evidence = retriever.retrieve(
        claim,
        k=k_evidence,
    )

    # =========================================================
    # 2. CREATE ADVOCATES
    # =========================================================

    for_agent = AdvocateAgent(
        llm,
        stance="FOR",
    )

    against_agent = AdvocateAgent(
        llm,
        stance="AGAINST",
    )

    # =========================================================
    # 3. OPENING ARGUMENTS
    # =========================================================

    for_argument = for_agent.argue(
        claim,
        evidence,
    )

    against_argument = against_agent.argue(
        claim,
        evidence,
    )

    # =========================================================
    # 4. REBUTTAL ROUND
    # =========================================================

    for_rebuttal = None
    against_rebuttal = None

    if rebuttal_round:

        # FOR responds to AGAINST
        for_rebuttal = for_agent.argue(
            claim,
            evidence,
            opponent_argument=against_argument.text,
        )

        # AGAINST responds to FOR
        against_rebuttal = against_agent.argue(
            claim,
            evidence,
            opponent_argument=for_argument.text,
        )

    # =========================================================
    # 5. JUDGE
    # =========================================================

    judge = JudgeAgent()

    (
        verdict,
        for_grounding,
        against_grounding,
        for_rebuttal_grounding,
        against_rebuttal_grounding,
    ) = judge.judge(
        claim=claim,
        evidence=evidence,
        for_argument=for_argument,
        against_argument=against_argument,
        for_rebuttal=for_rebuttal,
        against_rebuttal=against_rebuttal,
    )

    # =========================================================
    # 6. RETURN COMPLETE DEBATE RESULT
    # =========================================================

    return DebateResult(
        claim=claim,
        evidence=evidence,

        # Opening arguments
        for_argument=for_argument,
        against_argument=against_argument,

        # Opening grounding
        for_grounding=for_grounding,
        against_grounding=against_grounding,

        # Final verdict
        verdict=verdict,

        # Rebuttals
        for_rebuttal=for_rebuttal,
        against_rebuttal=against_rebuttal,

        # Rebuttal grounding
        for_rebuttal_grounding=for_rebuttal_grounding,
        against_rebuttal_grounding=against_rebuttal_grounding,
    )