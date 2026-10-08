import json

from debate_agents.graph import build_debate_graph
from debate_agents.llm import DummyLLM
from debate_agents.retriever import FaissRetriever


with open("data/evidence_corpus.json", "r", encoding="utf-8") as f:
    corpus = json.load(f)


retriever = FaissRetriever(corpus)

graph = build_debate_graph(
    retriever=retriever,
    llm=DummyLLM(),
    rebuttal_round=True,
)


result = graph.invoke(
    {
        "claim": "Goldfish have a three second memory."
    }
)


print("\n========== LANGGRAPH RESULT ==========")

print("\nFOR:")
print(result["for_argument"].text)

print("\nAGAINST:")
print(result["against_argument"].text)

print("\nFOR REBUTTAL:")
print(result["for_rebuttal"].text)

print("\nAGAINST REBUTTAL:")
print(result["against_rebuttal"].text)

print("\nVERDICT:")
print(result["verdict"].verdict)

print("\nCONFIDENCE:")
print(result["verdict"].confidence)