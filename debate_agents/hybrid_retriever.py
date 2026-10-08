"""
hybrid_retriever.py
===================

V2 hybrid evidence retrieval.

Strategy:

1. Search the local FAISS corpus.
2. Check whether local evidence is sufficiently relevant.
3. If local evidence is insufficient, search the web.
4. Return a combined evidence pool.

The retriever does not make the final factual judgment.
"""

from typing import List

from .retriever import FaissRetriever
from .web_retriever import WebRetriever
from .schema import EvidenceSnippet


class HybridRetriever:
    def __init__(
        self,
        local_retriever: FaissRetriever,
        web_retriever: WebRetriever,
        local_k: int = 4,
        web_k: int = 5,
        local_threshold: float = 0.55,
    ):
        self.local_retriever = local_retriever
        self.web_retriever = web_retriever

        self.local_k = local_k
        self.web_k = web_k
        self.local_threshold = local_threshold

    def retrieve(
        self,
        query: str,
        k: int = 6,
    ) -> List[EvidenceSnippet]:

        # -----------------------------------------------------
        # 1. Search local FAISS
        # -----------------------------------------------------

        local_results = self.local_retriever.retrieve(
            query,
            k=self.local_k,
        )

        # -----------------------------------------------------
        # 2. Decide whether local evidence is sufficient
        # -----------------------------------------------------

        local_sufficient = False

        if local_results:
            top_score = local_results[0].score

            if top_score >= self.local_threshold:
                local_sufficient = True

        # -----------------------------------------------------
        # 3. Local-only path
        # -----------------------------------------------------

        if local_sufficient:
            return local_results[:k]

        # -----------------------------------------------------
        # 4. External web fallback
        # -----------------------------------------------------

        web_results = self.web_retriever.retrieve(
            query,
            k=self.web_k,
        )

        # -----------------------------------------------------
        # 5. Merge evidence
        # -----------------------------------------------------

        combined = []

        combined.extend(local_results)
        combined.extend(web_results)

        # -----------------------------------------------------
        # 6. Remove duplicate URLs/text
        # -----------------------------------------------------

        unique = []
        seen = set()

        for item in combined:

            identifier = (
                item.url
                if item.url
                else item.text[:200]
            )

            if identifier in seen:
                continue

            seen.add(identifier)
            unique.append(item)

        # -----------------------------------------------------
        # 7. Sort by retrieval score
        # -----------------------------------------------------

        unique.sort(
            key=lambda item: item.retrieval_score
            if item.retrieval_score
            else item.score,
            reverse=True,
        )

        return unique[:k]