"""
Hallucination / Faithfulness Eval for FinSight.

Answers the question every RAG interview asks: "how do you know your
system isn't making things up?"

Two checks run on every generated answer, using the exact context that
was retrieved for that query (no ground-truth labeling needed):

1. NUMERIC GROUNDING (fast, deterministic, no extra LLM call)
   Every number, dollar amount, or percentage the Synthesizer states in
   its answer must appear somewhere in the retrieved context. Financial
   answers are dense with figures, so this alone catches the most
   damaging class of hallucination: invented numbers.

2. LLM-JUDGE FAITHFULNESS (optional, one extra Groq call per question)
   A judge prompt asks the model to rate 0-100 whether every claim in
   the answer is supported by the context, and to list anything that
   isn't. Slower and costs tokens, so it's opt-in via --judge.

Usage:
    python -m eval.hallucination_eval --questions eval/golden_questions.json
    python -m eval.hallucination_eval --questions eval/golden_questions.json --judge

golden_questions.json format:
    [{"question": "What was Apple's total net sales in fiscal 2024?"}, ...]
(Requires a document already indexed via the FinSight UI or index_document().)
"""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.pipeline import FinSightPipeline  # noqa: E402

NUMBER_PATTERN = re.compile(
    r"\$?\d[\d,]*\.?\d*\s?(?:%|percent|million|billion|thousand)?"
)

JUDGE_PROMPT = """You are auditing an AI-generated financial answer for hallucination.

Context (the ONLY source of truth the answer is allowed to use):
{context}

Generated Answer:
{answer}

Score how faithful the answer is to the context, 0-100, where 100 means
every claim is directly supported and 0 means the answer is fabricated.
Respond in EXACT format:

SCORE: [0-100]
UNSUPPORTED_CLAIMS: [list any claims not backed by the context, or "None"]"""


def extract_numbers(text: str) -> List[str]:
    """Pull normalized numeric tokens out of a string (strip commas/$/%)."""
    raw = NUMBER_PATTERN.findall(text)
    cleaned = []
    for tok in raw:
        digits = re.sub(r"[^\d.]", "", tok)
        if digits and digits not in (".",):
            cleaned.append(digits)
    return cleaned


def numeric_grounding_score(answer: str, context: str) -> Dict:
    """Fraction of numbers in the answer that also appear in the context."""
    answer_numbers = extract_numbers(answer)
    context_numbers = set(extract_numbers(context))

    if not answer_numbers:
        return {"grounded": 0, "total": 0, "score": None, "ungrounded_numbers": []}

    ungrounded = [n for n in answer_numbers if n not in context_numbers]
    grounded_count = len(answer_numbers) - len(ungrounded)

    return {
        "grounded": grounded_count,
        "total": len(answer_numbers),
        "score": round(grounded_count / len(answer_numbers), 3),
        "ungrounded_numbers": sorted(set(ungrounded)),
    }


def llm_judge_score(pipeline: FinSightPipeline, answer: str, context: str) -> Dict:
    prompt = JUDGE_PROMPT.format(context=context[:6000], answer=answer)
    response = pipeline.llm.generate_content(prompt)
    raw = response.text.strip()

    score_match = re.search(r"SCORE:\s*(\d+)", raw)
    claims_match = re.search(r"UNSUPPORTED_CLAIMS:\s*(.*)", raw, re.DOTALL)

    return {
        "score": int(score_match.group(1)) if score_match else None,
        "unsupported_claims": claims_match.group(1).strip() if claims_match else "Unknown",
    }


def run_eval(pipeline: FinSightPipeline, questions: List[dict], use_judge: bool) -> Dict:
    results = []

    for item in questions:
        question = item["question"]
        result = pipeline.query(question)

        # Reconstruct the context that was actually retrieved for this
        # query so grounding is checked against real evidence, not the
        # whole corpus.
        _, context = pipeline.retriever.run(result["sub_questions"])

        numeric = numeric_grounding_score(result["answer"], context)
        entry = {
            "question": question,
            "answer": result["answer"],
            "confidence": result["confidence"],
            "numeric_grounding": numeric,
        }

        if use_judge:
            entry["llm_judge"] = llm_judge_score(pipeline, result["answer"], context)

        results.append(entry)

    numeric_scores = [r["numeric_grounding"]["score"] for r in results if r["numeric_grounding"]["score"] is not None]
    summary = {
        "questions_evaluated": len(results),
        "avg_numeric_grounding_score": round(sum(numeric_scores) / len(numeric_scores), 3) if numeric_scores else None,
    }
    if use_judge:
        judge_scores = [r["llm_judge"]["score"] for r in results if r.get("llm_judge", {}).get("score") is not None]
        summary["avg_llm_judge_score"] = round(sum(judge_scores) / len(judge_scores), 1) if judge_scores else None

    return {"summary": summary, "results": results}


def main():
    parser = argparse.ArgumentParser(description="FinSight hallucination/faithfulness eval")
    parser.add_argument("--questions", required=True, help="Path to golden_questions.json")
    parser.add_argument("--judge", action="store_true", help="Also run the LLM-judge faithfulness check")
    parser.add_argument("--out", default="eval/eval_report.json", help="Where to write the report")
    args = parser.parse_args()

    questions = json.loads(Path(args.questions).read_text())
    pipeline = FinSightPipeline()

    report = run_eval(pipeline, questions, use_judge=args.judge)

    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["summary"], indent=2))
    print(f"\nFull report written to {args.out}")


if __name__ == "__main__":
    main()
