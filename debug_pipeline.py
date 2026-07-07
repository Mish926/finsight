"""
Debug script: runs the REAL pipeline (Planner -> Retriever -> Critic ->
Synthesizer) for one question and prints what each agent actually saw,
including the full (untruncated) context sent to the LLM. Uses your real
GROQ_API_KEY and makes real API calls.

Run from the finsight project root:
    PYTHONPATH=. python debug_pipeline.py
"""

from core.pipeline import FinSightPipeline

QUESTION = "What was JPMorgan's total revenue for fiscal year 2025?"

pipeline = FinSightPipeline()

if pipeline.vector_store.is_empty():
    print("No documents indexed -- upload documents first.")
    exit()

print(f"Question: {QUESTION!r}\n")

print("[1/4] Planner...")
sub_questions = pipeline.planner.run(QUESTION)
print("  Sub-questions:", sub_questions)

print("\n[2/4] Retriever...")
results, context = pipeline.retriever.run(sub_questions)
print(f"  Retrieved {len(results)} chunks, context length = {len(context)} chars\n")
print("  --- FULL CONTEXT SENT TO CRITIC/SYNTHESIZER ---")
print(context)
print("  --- END CONTEXT ---\n")

print("[3/4] Critic...")
verdict = pipeline.critic.run(QUESTION, context, results)
print(f"  sufficient={verdict.sufficient}  confidence={verdict.confidence}")
print(f"  issues={verdict.issues!r}")
print(f"  missing={verdict.missing!r}")
print(f"  verdict={verdict.verdict!r}")

print("\n[4/4] Synthesizer...")
result = pipeline.synthesizer.run(QUESTION, context, results, verdict, sub_questions)
print("  --- FINAL ANSWER ---")
print(result.answer)
