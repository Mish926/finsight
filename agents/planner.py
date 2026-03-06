"""
Agent 1: Query Planner
Decomposes a complex financial question into focused sub-questions.
This is what separates a naive RAG from an agentic system.
"""


import json
import re
from typing import List


PLANNER_PROMPT = """You are a financial research analyst. Decompose the following 
financial question into 2-4 specific, focused sub-questions that together will 
answer the original question completely.

Rules:
- Each sub-question must be answerable from a financial document (10-K, earnings report)
- Sub-questions should be specific and factual, not vague
- Avoid redundancy — each sub-question should cover a distinct aspect
- Return ONLY a JSON array of strings, no explanation, no markdown

Question: {question}
Output:"""


class QueryPlannerAgent:
    def __init__(self, model):
        self.model = model
        self.name = "QueryPlanner"

    def run(self, question: str) -> List[str]:
        """Decompose question into sub-questions."""
        prompt = PLANNER_PROMPT.format(question=question)

        try:
            response = self.model.generate_content(prompt)
            raw = response.text.strip()

            # Strip markdown code fences if present
            raw = re.sub(r'```json|```', '', raw).strip()

            match = re.search(r'\[.*\]', raw, re.DOTALL)
            if match:
                sub_questions = json.loads(match.group())
                if isinstance(sub_questions, list) and all(
                    isinstance(q, str) for q in sub_questions
                ):
                    return sub_questions[:4]

        except Exception as e:
            print(f"[QueryPlanner] Decomposition failed: {e}")

        return [question]
