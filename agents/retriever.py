"""
Agent 2: Retriever
Runs semantic search for each sub-question and deduplicates results.
"""

from typing import List, Tuple
from core.document_processor import Chunk
from core.vector_store import VectorStore


class RetrieverAgent:
    def __init__(self, vector_store: VectorStore, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k
        self.name = "Retriever"

    def run(self, sub_questions: List[str]) -> Tuple[List[Tuple[Chunk, float, str]], str]:
        """
        Search for each sub-question, deduplicate, return:
        - results: list of (chunk, score, sub_question)
        - context: formatted string for LLM
        """
        seen_ids = set()
        all_results = []

        for question in sub_questions:
            try:
                hits = self.vector_store.search(question, top_k=self.top_k)
                for chunk, score in hits:
                    if chunk.chunk_id not in seen_ids:
                        seen_ids.add(chunk.chunk_id)
                        all_results.append((chunk, score, question))
            except Exception as e:
                print(f"[Retriever] Search failed for '{question}': {e}")

        # Sort by score descending
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Build formatted context string
        context = self._format_context(all_results)

        return all_results, context

    def _format_context(self, results: List[Tuple[Chunk, float, str]]) -> str:
        if not results:
            return "No relevant content found."

        parts = []
        for i, (chunk, score, question) in enumerate(results, 1):
            parts.append(
                f"[Source {i} | {chunk.source} | Page {chunk.page} | "
                f"Relevance: {score:.2f}]\n{chunk.text}"
            )

        return "\n\n---\n\n".join(parts)
