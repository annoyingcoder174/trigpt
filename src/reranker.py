# src/reranker.py

from __future__ import annotations

from typing import List, Tuple

import torch
from sentence_transformers import CrossEncoder


class Reranker:
    """
    Cross-encoder based reranker.
    Given a query and candidate docs, returns them sorted by relevance.
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        if torch.backends.mps.is_available():
            self.device = "mps"
            print("Using MPS (Apple Metal) for reranker 🚀")
        else:
            self.device = "cpu"
            print("Using CPU for reranker")

        # CrossEncoder handles device internally via the `device` arg
        self.model = CrossEncoder(model_name, device=self.device)

    def rerank(
        self,
        query: str,
        docs: List[str],
        top_k: int | None = None,
    ) -> Tuple[List[int], List[float]]:
        """
        Returns (indices, scores), where indices are sorted in descending score.
        """
        if not docs:
            return [], []

        pairs = [(query, d) for d in docs]
        scores = self.model.predict(pairs)  # shape: (len(docs),)

        # sort indices by score desc
        indices = list(range(len(docs)))
        indices.sort(key=lambda i: float(scores[i]), reverse=True)

        if top_k is not None:
            indices = indices[:top_k]

        sorted_scores = [float(scores[i]) for i in indices]
        return indices, sorted_scores
