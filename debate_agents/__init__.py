"""
debate_agents
=============

A multi-agent adversarial fact-checking system.

The core idea (this is the concept to understand, not just the code):
    Instead of asking ONE model "is this claim true?" (which just gives you
    its gut reaction), we make TWO agents ARGUE opposite sides, forcing each
    one to dig up and cite real evidence. A separate JUDGE agent then scores
    each side not on how persuasive they *sound*, but on how well their
    claims are actually *grounded* in the evidence they cited.

    This is inspired by real AI-safety research on "debate" as a scalable
    oversight technique (get two systems to argue so a weaker judge can
    catch a stronger arguer being deceptive) and by fact-checking research
    on evidence-grounded verification.

Pipeline:
    claim -> Retriever -> [Advocate FOR, Advocate AGAINST] -> Judge -> verdict

Package layout:
    schema.py     - data structures passed between stages (the "state")
    retriever.py  - pulls relevant evidence snippets for a claim
    llm.py        - swappable LLM backends (Dummy for testing, Gemini for real use)
    agents.py     - the Advocate and Judge agent classes
    rubric.py     - the scoring rubric + programmatic grounding checks
    debate.py     - orchestrates the whole pipeline end-to-end
"""
