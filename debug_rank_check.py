"""
Diagnostic: find out EXACTLY where JPMorgan's real "Selected income
statement data" summary chunk ranks for a net-income/revenue query,
directly against your live index -- not a synthetic reproduction.

Run from the finsight project root:
    PYTHONPATH=. python debug_rank_check.py
"""

from core.vector_store import VectorStore

vs = VectorStore(index_dir="data/index")
if not vs.load():
    print("No index found -- upload documents first.")
    exit()

jpm_source = next((s for s in {c.source for c in vs.chunks} if "annualreport" in s.lower()), None)
if not jpm_source:
    print("Could not find a JPMorgan-like source in the index.")
    exit()

# 1) Find the actual chunk(s) containing the real headline figures, by
#    brute-force scanning for the exact numbers we verified externally.
print("=== Brute-force scan: chunks containing JPMorgan's real 2025 figures ===")
target_numbers = ["182,447", "57,048"]
found_any = False
for c in vs.chunks:
    if c.source == jpm_source and any(n in c.text for n in target_numbers):
        found_any = True
        print(f"FOUND at chunk_id={c.chunk_id}, page={c.page}, is_table={getattr(c, 'is_table', False)}")
        print(f"  text preview: {c.text[:200]!r}")
if not found_any:
    print("Not found anywhere in the index for this source -- the chunk containing")
    print("the real figures may not exist in the corpus at all (extraction gap),")
    print("rather than a ranking problem.")

# 2) If found, check where it ranks for the actual query
print()
print("=== Ranking check: where does that chunk rank for a real query? ===")
query = "JPMorgan total revenue net income fiscal year 2025"
results = vs.search(query, top_k=len(vs.chunks), source=jpm_source)  # rank ALL jpm chunks
for rank, (chunk, score) in enumerate(results, 1):
    if any(n in chunk.text for n in target_numbers):
        print(f"Real answer chunk ranks #{rank} out of {len(results)} JPMorgan chunks (score={score:.4f})")
        break
else:
    print("Real answer chunk did not appear in search results at all.")

print()
print("Top 5 ranked JPMorgan chunks for this query (for comparison):")
for rank, (chunk, score) in enumerate(results[:5], 1):
    print(f"  #{rank} score={score:.4f} page={chunk.page}  {chunk.text[:80]!r}")
