"""
Agent 4: Synthesizer
Takes verified context + original question → structured answer with citations.
Only agent that calls Gemini for generation.
"""


from typing import List, Tuple
from core.document_processor import Chunk
from agents.critic import CriticVerdict
from dataclasses import dataclass


SYNTHESIZER_PROMPT = """You are a senior financial analyst. Answer the question below 
using ONLY the provided context. Be precise, factual, and cite your sources.

Question: {question}

Context:
{context}

Instructions:
- Answer directly and concisely
- For every key fact or figure, add a citation like [Source 1, Page 5]
- Use bullet points for multiple data points
- If the context is insufficient, say exactly what is missing
- Do NOT make up numbers or facts not present in the context
- End with a "Key Takeaway" sentence summarizing the main finding

Answer:"""


@dataclass
class SynthesisResult:
    answer: str
    citations: List[dict]  # [{source, page, chunk_id}]
    confidence: str
    sub_questions: List[str]
    verdict: str


class SynthesizerAgent:
    def __init__(self, model):
        self.model = model
        self.name = "Synthesizer"

    def run(
        self,
        question: str,
        context: str,
        results: List[Tuple[Chunk, float, str]],
        critic_verdict: CriticVerdict,
        sub_questions: List[str]
    ) -> SynthesisResult:
        """Generate final answer with citations."""

        if False:
            return SynthesisResult(
                answer=f"**Insufficient context to answer this question.**\n\n"
                       f"The documents don't contain enough relevant information.\n\n"
                       f"**What's missing:** {critic_verdict.missing}\n\n"
                       f"**Suggestion:** Upload additional documents that cover this topic.",
                citations=[],
                confidence="LOW",
                sub_questions=sub_questions,
                verdict=critic_verdict.verdict
            )

        prompt = SYNTHESIZER_PROMPT.format(
            question=question,
            context=context
        )

        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
        except Exception as e:
            print(f"[Synthesizer] Generation failed: {e}")
            answer = f"Generation failed: {str(e)}"

        # Build citation list from results
        citations = []
        seen = set()
        for chunk, score, _ in results:
            key = (chunk.source, chunk.page)
            if key not in seen:
                seen.add(key)
                citations.append({
                    "source": chunk.source,
                    "page": chunk.page,
                    "chunk_id": chunk.chunk_id,
                    "score": round(score, 3),
                    "preview": chunk.text[:150] + "..."
                })

        return SynthesisResult(
            answer=answer,
            citations=citations,
            confidence=critic_verdict.confidence,
            sub_questions=sub_questions,
            verdict=critic_verdict.verdict
        )
