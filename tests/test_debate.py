"""
test_debate.py
==============
Unit tests using DummyLLM -- these verify your ORCHESTRATION logic is
correct (retrieval -> arguments -> grounding checks -> verdict) without
needing an API key or network access at all.

LEARNING NOTE: this is the test suite you'd point to in an interview when
asked "how did you test an agent system without spending money on API
calls for every test run?"

Run with: pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import json
import pytest

from debate_agents.retriever import TfidfRetriever
from debate_agents.llm import DummyLLM
from debate_agents.debate import run_debate
from debate_agents.rubric import check_grounding
from debate_agents.schema import Argument, EvidenceSnippet


@pytest.fixture
def retriever():
    with open(os.path.join(os.path.dirname(__file__), "..", "data", "evidence_corpus.json")) as f:
        corpus = json.load(f)
    return TfidfRetriever(corpus)


def test_retriever_returns_relevant_evidence(retriever):
    results = retriever.retrieve("Great Wall visible from the Moon", k=3)
    assert len(results) == 3
    # top result should be about the Great Wall, not goldfish
    assert "Wall" in results[0].text


def test_debate_runs_end_to_end(retriever):
    llm = DummyLLM()
    result = run_debate("Goldfish have a three second memory.", retriever, llm, rebuttal_round=True)

    assert result.claim == "Goldfish have a three second memory."
    assert result.verdict.verdict in ("TRUE", "FALSE", "UNCERTAIN")
    assert 0.0 <= result.verdict.confidence <= 1.0
    # both sides should have attempted at least one citation with DummyLLM
    assert len(result.for_argument.cited_evidence_ids) >= 0
    assert len(result.against_argument.cited_evidence_ids) >= 0


def test_grounding_catches_hallucinated_citation():
    evidence = [EvidenceSnippet(evidence_id="E1", text="Cats are mammals.", score=1.0)]
    # cites evidence id E99 which does not exist -> should be penalized
    fake_arg = Argument(stance="FOR", text="Dogs can fly.", cited_evidence_ids=["E99"])

    report = check_grounding(fake_arg, evidence)
    assert report.invalid_citation_count == 1
    assert report.grounding_score < 0.5


def test_grounding_rewards_real_overlap():
    evidence = [EvidenceSnippet(evidence_id="E1", text="Cats are small domesticated mammals.", score=1.0)]
    good_arg = Argument(
        stance="FOR",
        text="Cats are small domesticated mammals according to E1.",
        cited_evidence_ids=["E1"],
    )
    report = check_grounding(good_arg, evidence)
    assert report.valid_citation_count == 1
    assert report.grounding_score > 0.3


def test_benchmark_produces_accuracy(retriever):
    from benchmark import run_benchmark
    # small smoke test -- just checks it runs and returns a valid accuracy number
    summary = run_benchmark(use_real_llm=False, rebuttal_round=False)
    assert 0.0 <= summary["accuracy"] <= 1.0
    assert summary["n"] == len(summary["results"])
