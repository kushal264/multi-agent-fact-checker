"""
retriever.py
============
Semantic evidence retrieval using Sentence Transformers + FAISS.

Pipeline:

claim
  ↓
SentenceTransformer embedding
  ↓
FAISS similarity search
  ↓
top-k evidence snippets
"""

from typing import List

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .schema import EvidenceSnippet


class FaissRetriever:
    def __init__(
        self,
        corpus: List[dict],
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        """
        corpus:
            [
                {"id": "E1", "text": "..."},
                {"id": "E2", "text": "..."},
                ...
            ]
        """

        self.corpus = corpus

        self.ids = [item["id"] for item in corpus]
        self.texts = [item["text"] for item in corpus]

        # Load embedding model once.
        self.model = SentenceTransformer(model_name)

        # Create document embeddings.
        embeddings = self.model.encode(
            self.texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        embeddings = embeddings.astype("float32")

        self.dimension = embeddings.shape[1]

        # Inner-product search on normalized vectors
        # behaves like cosine similarity.
        self.index = faiss.IndexFlatIP(self.dimension)

        self.index.add(embeddings)

    def retrieve(
        self,
        query: str,
        k: int = 4,
    ) -> List[EvidenceSnippet]:

        # Encode query.
        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        query_embedding = query_embedding.astype("float32")

        # FAISS search.
        scores, indices = self.index.search(query_embedding, k)

        results = []

        for score, index in zip(scores[0], indices[0]):

            # FAISS returns -1 when there isn't a result.
            if index == -1:
                continue

            results.append(
                EvidenceSnippet(
                    evidence_id=self.ids[index],
                    text=self.texts[index],
                    score=float(score),
                )
            )

        return results

    def get_text(self, evidence_id: str) -> str:
        """
        Look up raw evidence by ID.
        Used by grounding checks.
        """

        for eid, text in zip(self.ids, self.texts):
            if eid == evidence_id:
                return text

        return ""
    # Backward compatibility for existing tests/benchmark code.
# The implementation is now FAISS-based.
TfidfRetriever = FaissRetriever