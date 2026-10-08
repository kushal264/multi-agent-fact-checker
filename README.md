# Multi-Agent Adversarial Fact-Checking System

Two AI agents argue opposite sides of a claim, citing real retrieved evidence.
A Judge agent scores both sides using a **programmatic grounding check**
(not just "which argument sounds more convincing") and issues a verdict.

```
claim
  │
  ▼
Retriever (TF-IDF over evidence corpus)
  │
  ├──► Advocate FOR ──┐
  │                    ├─► (optional rebuttal round) ─► Judge ─► verdict
  └──► Advocate AGAINST┘
```

---

## Why this design (the concepts to actually learn)

**1. Debate as a verification mechanism, not just "more agents."**
A single LLM asked "is this true?" gives you its gut reaction. Forcing two
agents to argue opposite sides and cite evidence surfaces the strongest
case *and* the strongest counter-case, which a Judge can then compare. This
mirrors real AI-safety research on debate as a scalable oversight technique.

**2. Never trust the Judge's self-report — check citations in code.**
`rubric.py` has zero LLM calls. It checks: does each cited evidence id
actually exist? Does the argument's text actually overlap with what that
evidence says? This is what stops a persuasive-but-wrong argument from
winning just because it *sounds* confident. The LLM Judge only supplies a
"logical consistency" score on top of this — it never gets the final say
by itself. **This is the single most important idea in the project.** Most
"LLM as judge" demos skip this and are gameable; yours isn't.

**3. Agents depend on interfaces, not concrete providers.**
`agents.py` never imports Gemini directly — it calls `llm.generate(prompt)`
on whatever `BaseLLM` it's given. Same for the retriever. This is why you
can test the entire pipeline with `DummyLLM` (zero cost, zero network,
fully deterministic) and swap in `GeminiLLM` later by changing one line.
This pattern (dependency inversion) is what makes real production agent
systems maintainable.

**4. Shared state, not shared memory.**
`schema.py` defines exactly what data passes between agents (claim →
evidence → arguments → grounding reports → verdict). Every agent reads
from and writes to this shape. If you later port this to LangGraph, this
dataclass shape becomes the graph's `State` almost unchanged — the
orchestration plumbing changes, the state design doesn't.

---

## Project layout

```
debate_agents/
  schema.py     — shared data structures (the "state")
  llm.py        — swappable LLM backend: DummyLLM (offline) / GeminiLLM (real)
  retriever.py  — TF-IDF evidence retrieval (same concept as your FAISS work)
  rubric.py     — programmatic grounding checker + scoring
  agents.py     — AdvocateAgent, JudgeAgent
  debate.py     — orchestrates the full pipeline
data/
  evidence_corpus.json  — small knowledge base to retrieve from
  claims.json            — benchmark claims with TRUE/FALSE ground-truth labels
tests/
  test_debate.py — unit tests using DummyLLM, no API key needed
benchmark.py     — runs all claims, prints accuracy
api.py           — FastAPI service (matches your Face Verification / Plant
                    Disease projects' deployment style)
```

---

## Running it

```bash
pip install -r requirements.txt

# 1. Run tests (fully offline, uses DummyLLM)
pytest tests/ -v

# 2. Run the benchmark offline (sanity check the plumbing)
python benchmark.py --no-rebuttal

# 3. Run the REAL system with Gemini
pip install google-generativeai
cp .env.example .env   # fill in GEMINI_API_KEY
export $(cat .env | xargs)
python benchmark.py --real

# 4. Serve it as an API (production style, like your other projects)
uvicorn api:app --reload
curl -X POST localhost:8000/debate -H "Content-Type: application/json" \
     -d '{"claim": "Goldfish have a three second memory."}'
```

### Why the DummyLLM benchmark shows 0% / all "UNCERTAIN"

This is expected, not a bug. `DummyLLM` is a naive rule-based stub — it
doesn't actually reason about the claim, so its FOR and AGAINST arguments
end up roughly equally (un)grounded, and the Judge correctly reports
`UNCERTAIN` rather than faking confidence it doesn't have. That's the
grounding check doing its job. Real accuracy numbers only mean something
once you run `--real` with an actual LLM. **Put the `--real` accuracy
number on your resume, not the DummyLLM one.**

---

## What to actually do next (in priority order)

1. **Run it for real.** Get a free Gemini API key, run `python benchmark.py --real`,
   note the accuracy — that's your resume metric (e.g. "achieved X% accuracy
   on an 8-claim adversarial fact-checking benchmark").
2. **Grow the benchmark.** 8 claims is a demo, not a benchmark. Add 30-50
   claims across a few categories (science myths, statistics, historical
   claims) with real evidence snippets, and report accuracy per category.
3. **Swap TF-IDF for FAISS + embeddings** in `retriever.py` — you already
   know how from your RAG project. Keep the same `.retrieve()` interface
   and nothing else needs to change.
4. **Containerize it.** Add a Dockerfile around `api.py`, same pattern as
   your Face Verification and Plant Disease projects — this is what keeps
   the project consistent with the rest of your production-flavored resume.
5. **(Stretch) Port `debate.py` to LangGraph.** Once you're comfortable
   with the plain-Python version, rewrite `run_debate()` as a LangGraph
   graph with `State = schema.DebateResult`-shaped TypedDict and nodes for
   each step. This is a good way to learn LangGraph's mental model because
   you'll already know exactly what each node should do.

---

## Interview talking points this project gives you

- "Why debate instead of a single-agent classifier?" → adversarial framing
  surfaces the strongest argument on both sides, which a single-pass
  classifier can't do.
- "How do you stop the Judge from just picking whichever argument sounds
  more confident?" → programmatic grounding check runs first, in plain
  code, before any LLM opinion is even asked for.
- "How did you test this without burning API budget?" → `DummyLLM` lets
  you unit test 100% of the orchestration logic offline.
- "What would you improve with more time?" → bigger benchmark, real
  embeddings via FAISS, multi-round rebuttals with diminishing returns
  tracked, calibration of the confidence score against real accuracy.
