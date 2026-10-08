"""
web_retriever.py
================

External web retrieval for the V2 open-world fact checker.

The retriever is intentionally separate from the FAISS retriever.

V2 architecture:

Claim
  ↓
WebRetriever
  ↓
External web evidence
  ↓
EvidenceSnippet objects
"""

import os
from typing import List
from urllib.parse import urlparse

from tavily import TavilyClient

from .schema import EvidenceSnippet


class WebRetriever:
    """
    Retrieve external evidence using Tavily.

    This class does NOT decide whether the evidence is sufficient.
    That will be handled later by the evidence-sufficiency gate.
    """

    def __init__(
        self,
        api_key: str | None = None,
        max_results: int = 5,
        include_raw_content: bool = True,
    ):
        self.api_key = (
            api_key
            or os.environ.get("TAVILY_API_KEY")
        )

        self.max_results = max_results
        self.include_raw_content = include_raw_content

        if self.api_key:
            self.client = TavilyClient(
                api_key=self.api_key
            )
        else:
            # Tavily currently supports keyless search,
            # but this mode is rate-limited.
            self.client = TavilyClient()

    def retrieve(
        self,
        query: str,
        k: int | None = None,
    ) -> List[EvidenceSnippet]:

        limit = k or self.max_results

        response = self.client.search(
            query=query,
            search_depth="advanced",
            max_results=limit,
            include_raw_content=(
                "text"
                if self.include_raw_content
                else False
            ),
        )

        results = []

        for index, item in enumerate(
            response.get("results", []),
            start=1,
        ):
            url = item.get("url", "")
            title = item.get("title", "")

            # Prefer extracted article content.
            text = item.get(
                "raw_content"
            ) or item.get(
                "content",
                "",
            )

            if not text:
                continue

            # Avoid sending enormous pages into the LLM.
            text = text.strip()[:5000]

            domain = urlparse(url).netloc

            retrieval_score = float(
                item.get("score", 0.0) or 0.0
            )

            published_date = item.get(
                "published_date"
            )

            evidence_id = f"W{index}"

            results.append(
                EvidenceSnippet(
                    evidence_id=evidence_id,
                    text=text,
                    score=retrieval_score,
                    source_type="web",
                    title=title,
                    url=url,
                    domain=domain,
                    published_date=published_date,
                    retrieval_score=retrieval_score,
                    credibility_score=0.0,
                )
            )

        return results