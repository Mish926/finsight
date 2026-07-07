"""
Debug script: shows exactly what's in your index for Apple, and how it
ranks against a revenue query. Run from the finsight project root:

    PYTHONPATH=. python debug_search.py
"""

from core.vector_store import VectorStore

vs = VectorStore(index_dir="data/index")
if not vs.load():
    print("No index found at data/index/finsight.pkl -- upload documents first.")
    exit()

print(f"\nTotal chunks in index: {len(vs.chunks)}")
sources = {}
for c in vs.chunks:
    sources[c.source] = sources.get(c.source, 0) + 1
print("Chunks per source:", sources)

# 1) Search ONLY within the Apple document for revenue-related terms,
#    with a generous top_k so we can see deep into the ranking.
print("\n=== Top 20 Apple-only results for 'total revenue net sales fiscal 2024' ===")
apple_source = next((s for s in sources if "apple" in s.lower()), None)
if apple_source:
    results = vs.search("total revenue net sales fiscal 2024", top_k=20, source=apple_source)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"{i:2d}. score={score:.4f}  page={chunk.page}  {chunk.text[:100]!r}")
else:
    print("Could not find an Apple source in the index.")

# 2) Brute-force scan: find any chunk (regardless of score) whose raw text
#    contains a real dollar-figure revenue mention, to confirm it's even
#    in the index at all and see its exact wording/page.
print("\n=== Brute-force scan: Apple chunks mentioning 'net sales' AND a $ figure ===")
found = False
for c in vs.chunks:
    if apple_source and c.source == apple_source and "net sales" in c.text.lower() and "$" in c.text:
        found = True
        print(f"page={c.page}  chunk_id={c.chunk_id}  {c.text[:200]!r}")
        print("---")
if not found:
    print("No chunk found containing both 'net sales' and a '$' figure for Apple.")
    print("This would mean the income statement table either didn't extract cleanly")
    print("from the PDF, or got split in a way that separates the label from the number.")
