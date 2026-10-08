from typing import Any, TypedDict

from langgraph.graph import StateGraph, START, END

from .agents import AdvocateAgent, JudgeAgent
from .schema import (
    Argument,
    DebateResult,
    EvidenceSnippet,
    GroundingReport,
)


class DebateState(TypedDict, total=False):
    claim: str

    evidence: list[EvidenceSnippet]

    for_argument: Argument
    against_argument: Argument

    for_rebuttal: Argument
    against_rebuttal: Argument

    for_grounding: GroundingReport
    against_grounding: GroundingReport

    for_rebuttal_grounding: GroundingReport
    against_rebuttal_grounding: GroundingReport

    verdict: Any

    result: DebateResult


def build_debate_graph(
    retriever,
    llm,
    k_evidence: int = 4,
    rebuttal_round: bool = True,
):
    """
    Build the LangGraph version of the debate pipeline.

    Dependencies such as the retriever and LLM are captured
    by the node closures instead of being stored in graph state.
    """

    for_agent = AdvocateAgent(
        llm,
        stance="FOR",
    )

    against_agent = AdvocateAgent(
        llm,
        stance="AGAINST",
    )

    judge = JudgeAgent()

    # =========================================================
    # NODE 1 — RETRIEVE EVIDENCE
    # =========================================================

    def retrieve_evidence(state: DebateState):
        evidence = retriever.retrieve(
            state["claim"],
            k=k_evidence,
        )

        return {
            "evidence": evidence,
        }

    # =========================================================
    # NODE 2 — FOR OPENING
    # =========================================================

    def for_opening(state: DebateState):
        argument = for_agent.argue(
            state["claim"],
            state["evidence"],
        )

        return {
            "for_argument": argument,
        }

    # =========================================================
    # NODE 3 — AGAINST OPENING
    # =========================================================

    def against_opening(state: DebateState):
        argument = against_agent.argue(
            state["claim"],
            state["evidence"],
        )

        return {
            "against_argument": argument,
        }

    # =========================================================
    # NODE 4 — FOR REBUTTAL
    # =========================================================

    def for_rebuttal(state: DebateState):
        if not rebuttal_round:
            return {}

        rebuttal = for_agent.argue(
            state["claim"],
            state["evidence"],
            opponent_argument=state["against_argument"].text,
        )

        return {
            "for_rebuttal": rebuttal,
        }

    # =========================================================
    # NODE 5 — AGAINST REBUTTAL
    # =========================================================

    def against_rebuttal(state: DebateState):
        if not rebuttal_round:
            return {}

        rebuttal = against_agent.argue(
            state["claim"],
            state["evidence"],
            opponent_argument=state["for_argument"].text,
        )

        return {
            "against_rebuttal": rebuttal,
        }

    # =========================================================
    # NODE 6 — JUDGE
    # =========================================================

    def judge_debate(state: DebateState):
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
            for_rebuttal=state.get("for_rebuttal"),
            against_rebuttal=state.get("against_rebuttal"),
        )

        result = DebateResult(
            claim=state["claim"],
            evidence=state["evidence"],

            for_argument=state["for_argument"],
            against_argument=state["against_argument"],

            for_grounding=for_grounding,
            against_grounding=against_grounding,

            verdict=verdict,

            for_rebuttal=state.get("for_rebuttal"),
            against_rebuttal=state.get("against_rebuttal"),

            for_rebuttal_grounding=for_rebuttal_grounding,
            against_rebuttal_grounding=against_rebuttal_grounding,
        )

        return {
            "verdict": verdict,
            "for_grounding": for_grounding,
            "against_grounding": against_grounding,
            "for_rebuttal_grounding": for_rebuttal_grounding,
            "against_rebuttal_grounding": against_rebuttal_grounding,
            "result": result,
        }

    # =========================================================
    # BUILD GRAPH
    # =========================================================

    builder = StateGraph(DebateState)

    builder.add_node(
        "retrieve_evidence",
        retrieve_evidence,
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

    # =========================================================
    # GRAPH EDGES
    # =========================================================

    builder.add_edge(
        START,
        "retrieve_evidence",
    )

    # Both advocates start after retrieval.
    builder.add_edge(
        "retrieve_evidence",
        "for_opening",
    )

    builder.add_edge(
        "retrieve_evidence",
        "against_opening",
    )

    # Rebuttals wait until BOTH opening arguments exist.
    builder.add_edge(
        ["for_opening", "against_opening"],
        "for_rebuttal",
    )

    builder.add_edge(
        ["for_opening", "against_opening"],
        "against_rebuttal",
    )

    # Judge waits until both rebuttal nodes complete.
    builder.add_edge(
        ["for_rebuttal", "against_rebuttal"],
        "judge",
    )

    builder.add_edge(
        "judge",
        END,
    )

    return builder.compile()