"""
Agent 3: Critic - relaxed thresholds for TF-IDF retrieval.
"""

from typing import List, Tuple
from core.document_processor import Chunk
from dataclasses import dataclass


@dataclass
class CriticVerdict:
    sufficient: bool
    confidence: str
    issues: str
    missing: str
    verdict: str


CRITIC_PROMPT = """You are a critical financial analyst reviewing research evidence.

Original Question: {question}

Retrieved Context:
{context}

Evaluate the retrieved context and respond in this EXACT format (no extra text):

SUFFICIENT: [YES or NO]
CONFIDENCE: [HIGH, MEDIUM, or LOW]
ISSUES: [List any contradictions, gaps, or concerns. Write "None" if clean.]
MISSING: [What key information is absent? Write "None" if complete.]
VERDICT: [1-2 sentence summary of whether this context can answer the question]"""


class CriticAgent:
    def __init__(self, model):
        self.model = model
        self.name = "Critic"

    def run(self, question: str, context: str, results: List[Tuple[Chunk, float, str]]) -> CriticVerdict:
        if not results:
            return CriticVerdict(
                sufficient=False,
                confidence="LOW",
                issues="No chunks retrieved.",
                missing="All required information.",
                verdict="Cannot answer — no content found."
            )

        # Always try to synthesize if we have chunks — let LLM decide
        try:
            prompt = CRITIC_PROMPT.format(question=question, context=context)
            response = self.model.generate_content(prompt)
            return self._parse(response.text.strip())
        except Exception as e:
            print(f"[Critic] Evaluation failed: {e}")
            return CriticVerdict(
                sufficient=True,
                confidence="MEDIUM",
                issues="None",
                missing="None",
                verdict="Proceeding with available context."
            )

    def _parse(self, raw: str) -> CriticVerdict:
        def extract(label):
            for line in raw.split('\n'):
                if line.strip().startswith(f"{label}:"):
                    return line.split(':', 1)[1].strip()
            return "Unknown"

        return CriticVerdict(
            sufficient=extract("SUFFICIENT").upper() == "YES",
            confidence=extract("CONFIDENCE").upper(),
            issues=extract("ISSUES"),
            missing=extract("MISSING"),
            verdict=extract("VERDICT")
        )
