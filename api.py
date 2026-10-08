"""
api.py
======

FastAPI backend for the V2 hybrid/open-world
multi-agent fact-checking system.

Architecture:

React
  ↓
FastAPI
  ↓
LangGraph
  ↓
Claim Classifier
  ↓
Hybrid Retriever
  ├── FAISS
  └── Web Search
  ↓
Evidence Validator
  ↓
Multi-Agent Debate
  ↓
Verdict

Run:

    uvicorn api:app --reload
"""

import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from debate_agents.retriever import FaissRetriever
from debate_agents.web_retriever import WebRetriever
from debate_agents.hybrid_retriever import HybridRetriever

from debate_agents.llm import (
    DummyLLM,
    GeminiLLM,
)

from debate_agents.graph import (
    build_debate_graph,
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Multi-Agent Open-World Fact Checker",
    version="2.0.0",
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# LOAD LOCAL EVIDENCE CORPUS
# ============================================================

with open(
    "data/evidence_corpus.json",
    "r",
    encoding="utf-8",
) as f:
    _corpus = json.load(f)


# ============================================================
# LOCAL FAISS RETRIEVER
# ============================================================

_local_retriever = FaissRetriever(
    _corpus
)


# ============================================================
# WEB RETRIEVER
# ============================================================

_web_retriever = WebRetriever(
    max_results=5,
    include_raw_content=True,
)


# ============================================================
# HYBRID RETRIEVER
# ============================================================

_hybrid_retriever = HybridRetriever(
    local_retriever=_local_retriever,
    web_retriever=_web_retriever,

    local_k=4,
    web_k=5,

    # Starting threshold.
    # We can calibrate this later.
    local_threshold=0.55,
)


# ============================================================
# LLM BACKEND
# ============================================================

if os.environ.get("GEMINI_API_KEY"):

    _llm = GeminiLLM()

else:

    _llm = DummyLLM()


# ============================================================
# BUILD LANGGRAPH INSTANCES
# ============================================================

# Build once instead of rebuilding the graph
# for every request.

_graph_with_rebuttal = build_debate_graph(
    retriever=_hybrid_retriever,
    llm=_llm,
    k_evidence=6,
    rebuttal_round=True,
)


_graph_without_rebuttal = build_debate_graph(
    retriever=_hybrid_retriever,
    llm=_llm,
    k_evidence=6,
    rebuttal_round=False,
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ClaimRequest(BaseModel):

    claim: str

    rebuttal_round: bool = True


# ============================================================
# HEALTH
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",

        "version": "2.0.0",

        "llm_backend": type(
            _llm
        ).__name__,

        "local_retriever": "FAISS",

        "web_retriever": "Tavily",

        "retriever": "Hybrid",

        "orchestration": "LangGraph",

        "mode": "open-world",
    }


# ============================================================
# DEBATE ENDPOINT
# ============================================================

@app.post("/debate")
def debate(req: ClaimRequest):

    claim = req.claim.strip()

    if not claim:

        return {
            "error": "Claim cannot be empty."
        }

    # --------------------------------------------------------
    # Select graph
    # --------------------------------------------------------

    if req.rebuttal_round:

        graph = _graph_with_rebuttal

    else:

        graph = _graph_without_rebuttal

    # --------------------------------------------------------
    # Execute LangGraph
    # --------------------------------------------------------

    state = graph.invoke(
        {
            "claim": claim,
        }
    )

    result = state["result"]

    # --------------------------------------------------------
    # Serialize evidence
    # --------------------------------------------------------

    evidence = [
        item.__dict__
        for item in result.evidence
    ]

    rejected_evidence = [
        item.__dict__
        for item in result.rejected_evidence
    ]

    # --------------------------------------------------------
    # Return complete V2 response
    # --------------------------------------------------------

    return {

        # ----------------------------------------------------
        # Claim
        # ----------------------------------------------------

        "claim": result.claim,

        "claim_type": result.claim_type,

        # ----------------------------------------------------
        # Retrieval
        # ----------------------------------------------------

        "retrieval_mode": (
            result.retrieval_mode
        ),

        "evidence_sufficient": (
            result.evidence_sufficient
        ),

        "evidence_reason": (
            result.evidence_reason
        ),

        # ----------------------------------------------------
        # Accepted evidence
        # ----------------------------------------------------

        "evidence": evidence,

        # ----------------------------------------------------
        # Rejected evidence
        # ----------------------------------------------------

        "rejected_evidence": (
            rejected_evidence
        ),

        # ----------------------------------------------------
        # Opening arguments
        # ----------------------------------------------------

        "for_argument": (
            result.for_argument.__dict__
        ),

        "against_argument": (
            result.against_argument.__dict__
        ),

        # ----------------------------------------------------
        # Grounding
        # ----------------------------------------------------

        "for_grounding": (
            result.for_grounding.__dict__
        ),

        "against_grounding": (
            result.against_grounding.__dict__
        ),

        # ----------------------------------------------------
        # Rebuttals
        # ----------------------------------------------------

        "for_rebuttal": (
            result.for_rebuttal.__dict__
            if result.for_rebuttal
            else None
        ),

        "against_rebuttal": (
            result.against_rebuttal.__dict__
            if result.against_rebuttal
            else None
        ),

        # ----------------------------------------------------
        # Rebuttal grounding
        # ----------------------------------------------------

        "for_rebuttal_grounding": (
            result.for_rebuttal_grounding.__dict__
            if result.for_rebuttal_grounding
            else None
        ),

        "against_rebuttal_grounding": (
            result.against_rebuttal_grounding.__dict__
            if result.against_rebuttal_grounding
            else None
        ),

        # ----------------------------------------------------
        # Verdict
        # ----------------------------------------------------

        "verdict": (
            result.verdict.__dict__
        ),
    }