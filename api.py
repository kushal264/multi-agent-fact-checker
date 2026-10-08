"""
api.py
======
FastAPI backend for the Multi-Agent Fact-Checking Debate System.

Architecture:

React
  ↓
FastAPI
  ↓
LangGraph
  ↓
FAISS + Multi-Agent Debate
  ↓
Verdict

Run:
    uvicorn api:app --reload

Endpoints:
    GET  /health
    POST /debate
"""

import json
import os

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from debate_agents.retriever import FaissRetriever
from debate_agents.llm import DummyLLM, GeminiLLM
from debate_agents.graph import build_debate_graph


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Multi-Agent Fact-Checking Debate System",
    version="1.0.0",
)


# ============================================================
# CORS
# ============================================================

# React development server normally runs on port 5173.
# FastAPI normally runs on port 8000.

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
# LOAD EVIDENCE CORPUS
# ============================================================

with open(
    "data/evidence_corpus.json",
    "r",
    encoding="utf-8",
) as f:
    _corpus = json.load(f)


# ============================================================
# BUILD FAISS RETRIEVER ONCE
# ============================================================

_retriever = FaissRetriever(_corpus)


# ============================================================
# SELECT LLM BACKEND
# ============================================================

if os.environ.get("GEMINI_API_KEY"):
    _llm = GeminiLLM()
else:
    _llm = DummyLLM()


# ============================================================
# BUILD LANGGRAPH ONCE
# ============================================================

_debate_graph = build_debate_graph(
    retriever=_retriever,
    llm=_llm,
    rebuttal_round=True,
)


# ============================================================
# REQUEST MODEL
# ============================================================

class ClaimRequest(BaseModel):
    claim: str
    rebuttal_round: bool = True


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_backend": type(_llm).__name__,
        "retriever": "FAISS",
        "orchestration": "LangGraph",
    }


# ============================================================
# DEBATE ENDPOINT
# ============================================================

@app.post("/debate")
def debate(req: ClaimRequest):

    # Build a graph for the requested rebuttal configuration.
    #
    # For normal frontend use, rebuttal_round=True is recommended.
    graph = build_debate_graph(
        retriever=_retriever,
        llm=_llm,
        rebuttal_round=req.rebuttal_round,
    )

    # Execute LangGraph.
    result = graph.invoke(
        {
            "claim": req.claim,
        }
    )

    debate_result = result["result"]

    return {
        # ----------------------------------------------------
        # Claim
        # ----------------------------------------------------

        "claim": debate_result.claim,

        # ----------------------------------------------------
        # Retrieved evidence
        # ----------------------------------------------------

        "evidence": [
            evidence.__dict__
            for evidence in debate_result.evidence
        ],

        # ----------------------------------------------------
        # Opening arguments
        # ----------------------------------------------------

        "for_argument": (
            debate_result.for_argument.__dict__
        ),

        "against_argument": (
            debate_result.against_argument.__dict__
        ),

        # ----------------------------------------------------
        # Opening grounding
        # ----------------------------------------------------

        "for_grounding": (
            debate_result.for_grounding.__dict__
        ),

        "against_grounding": (
            debate_result.against_grounding.__dict__
        ),

        # ----------------------------------------------------
        # Rebuttals
        # ----------------------------------------------------

        "for_rebuttal": (
            debate_result.for_rebuttal.__dict__
            if debate_result.for_rebuttal
            else None
        ),

        "against_rebuttal": (
            debate_result.against_rebuttal.__dict__
            if debate_result.against_rebuttal
            else None
        ),

        # ----------------------------------------------------
        # Rebuttal grounding
        # ----------------------------------------------------

        "for_rebuttal_grounding": (
            debate_result.for_rebuttal_grounding.__dict__
            if debate_result.for_rebuttal_grounding
            else None
        ),

        "against_rebuttal_grounding": (
            debate_result.against_rebuttal_grounding.__dict__
            if debate_result.against_rebuttal_grounding
            else None
        ),

        # ----------------------------------------------------
        # Final verdict
        # ----------------------------------------------------

        "verdict": (
            debate_result.verdict.__dict__
        ),
    }