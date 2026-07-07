"""
Agent 2: Retriever
Runs semantic search for each sub-question and deduplicates results.

When multiple documents are indexed (e.g. Apple + Amazon 10-Ks), a single
global top-k ranking can starve one document entirely if its chunks score
lower on TF-IDF — silently turning a comparison question into a single-
document answer. To avoid that, retrieval and context-capping are both
source-aware: each indexed document gets guaranteed representation.

Query expansion: different filers use different terminology for the same
line item -- Apple's 10-K says "net sales", Amazon's says "total revenue".
A user question phrased with one term can score the actual data row lower
than unrelated boilerplate that happens to repeat that term in a generic
policy-discussion sense (e.g. revenue recognition, deferred revenue). LSA
helps with this in theory, but on a modest single-filing corpus the
semantic space isn't always dense enough to bridge it reliably.

The synonym clusters below are standard GAAP/SEC financial-statement
terminology equivalences -- not specific to any one company -- so this
generalizes to any 10-K, 10-Q, or annual report a user uploads, not just
Apple/Amazon. Matching is symmetric: whichever term the question uses,
every other term in its cluster gets appended to the search query.
"""

import re
from collections import defaultdict, deque
from typing import List, Tuple
from core.document_processor import Chunk
from core.vector_store import VectorStore

# Groq's free tier caps requests at 6000 tokens/minute (org-wide, shared
# across prompt + completion). The original 12000-character budget assumed
# ~3.5-4 chars/token (typical English prose), but dense financial text --
# lots of "$", commas inside numbers, parenthesized figures -- tokenizes
# far less efficiently, closer to ~2.5 chars/token. A real request with
# ~12500 chars of context needed 6392 tokens total, over the 6000 cap, even
# with the completion reserved separately. 9000 chars leaves real safety
# margin at that actual density, and targeted retrieval (adaptive
# per-source probing, financial term expansion, summary-section boosting)
# now needs less raw volume to find the right chunk than when this budget
# was first set, so the tighter cap costs less than it did earlier.
MAX_CONTEXT_CHARS = 9000

# Standard GAAP/SEC filing terminology clusters. Each inner list is a set of
# interchangeable phrasings different filers use for the same line item.
# These are generic accounting terms, not tied to any specific company, so
# this expansion applies to any financial filing a user uploads -- retail,
# tech, banking, or otherwise.
FINANCIAL_TERM_CLUSTERS = [
    # Top-line revenue
    ["revenue", "revenues", "net sales", "net revenues", "total revenue",
     "total revenues", "total net sales", "sales", "total net revenue"],
    # Bottom-line profit
    ["net income", "net earnings", "net profit", "earnings", "profit",
     "income attributable to common shareholders"],
    ["operating income", "income from operations", "operating profit",
     "pre-provision profit"],
    ["gross profit", "gross margin"],
    # Costs and expenses
    ["operating expenses", "total costs and expenses", "opex",
     "noninterest expense", "total expenses"],
    ["cost of revenue", "cost of sales", "cost of goods sold", "cogs"],
    ["research and development", "r&d expense", "r&d"],
    ["selling general and administrative", "sg&a", "sga"],
    # Per-share and valuation
    ["earnings per share", "eps", "diluted eps", "basic eps"],
    ["book value", "book value per share", "tangible book value"],
    ["dividend", "dividends", "dividend per share", "dividends declared"],
    # Balance sheet
    ["total assets"],
    ["total liabilities"],
    ["stockholders equity", "shareholders equity", "total equity",
     "common stockholders equity"],
    # Cash flow and capital
    ["cash flow from operations", "operating cash flow", "free cash flow", "fcf"],
    ["capital expenditures", "capex", "purchases of property and equipment"],
    ["share repurchase", "share buyback", "stock repurchase", "buybacks"],
    # Banking / financial-institution specific (not applicable to retail/tech,
    # but essential for correctly answering questions on bank filings)
    ["net interest income", "nii"],
    ["noninterest income", "non-interest income", "fee income"],
    ["provision for credit losses", "provision for loan losses",
     "credit loss provision", "allowance for credit losses"],
    ["return on equity", "roe"],
    ["return on assets", "roa"],
    ["assets under management", "aum", "client assets"],
    ["deposits", "customer deposits", "total deposits"],
    ["loans", "loans and leases", "total loans", "loans retained"],
    ["tier 1 capital ratio", "common equity tier 1", "cet1"],
    # Subscriber/usage metrics (tech/media/telecom)
    ["subscribers", "subscriber count", "paid subscribers"],
    ["monthly active users", "mau", "daily active users", "dau"],
    ["same-store sales", "comparable sales", "comp sales", "same store sales"],
    ["backlog", "remaining performance obligations", "rpo"],
    # Period/reporting terminology
    ["fiscal year", "fiscal year end", "reporting period"],
    ["deferred revenue", "unearned revenue", "contract liability"],
]


# Detects chunks that look like a business-segment breakdown table --
# e.g. "Consumer & Community Banking", "Commercial & Investment Bank",
# "Asset & Wealth Management" all appearing together, JPMorgan's segment
# naming pattern. Multiple distinct segment-style labels in one chunk is
# a strong signal that it reports a SEGMENT's total, not the firm-wide
# one -- and this has been the repeated, confirmed root cause of
# aggregation-level mistakes across both FY2024 and FY2025 filings.
SEGMENT_LABEL_PATTERN = re.compile(r'\b[A-Z][a-z]+(?:\s*&\s*|\s+and\s+)[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b')
SEGMENT_QUESTION_PATTERN = re.compile(r'\bsegments?\b|\bby business\b|\bby division\b|\bbreakdown\b|\bbreak down\b', re.IGNORECASE)


class RetrieverAgent:
    def __init__(self, vector_store: VectorStore, top_k: int = 4):
        self.vector_store = vector_store
        self.top_k = top_k
        self.name = "Retriever"

    def _expand_query(self, question: str) -> str:
        """Append every synonym from any financial-term cluster matched in
        the question, so a chunk using different-but-equivalent wording
        (e.g. a different company's filing) can still be found. Matching
        is symmetric and uses word boundaries so short terms like "eps"
        don't false-match inside unrelated words."""
        lower_q = question.lower()
        expansions = set()
        for cluster in FINANCIAL_TERM_CLUSTERS:
            matched = any(
                re.search(r'\b' + re.escape(term) + r'\b', lower_q)
                for term in cluster
            )
            if matched:
                expansions.update(cluster)

        # A question asking for an overall/total headline figure ("total
        # revenue", "total net income"...) is best answered from a filing's
        # Summary/Highlights section -- e.g. "Three-Year Summary of
        # Consolidated Financial Highlights" -- which virtually every 10-K
        # has as the single cleanest, most authoritative source for exactly
        # this kind of number. Business-segment tables also legitimately
        # contain a "Total" figure for their own segment, and are often
        # structurally denser/harder to parse correctly (many segments x
        # many years in one wide table). Nudging retrieval toward the
        # summary-section wording increases the odds of landing on the
        # simple, unambiguous table instead of the complex segment one.
        if re.search(r'\btotal\b', lower_q) and expansions:
            expansions.update([
                "summary", "highlights", "selected financial data",
                "three-year summary", "financial highlights",
                "selected income statement data"
            ])

        # Similarly, a question about risk factors is best answered from
        # the actual "Item 1A: Risk Factors" section -- without this boost,
        # generic sentences that merely MENTION "risk" in passing (e.g.
        # forward-looking-statement boilerplate, which appears on many
        # pages) can outrank the real risk factor disclosures.
        if re.search(r'\brisks?\b', lower_q):
            expansions.update(["item 1a", "risk factors"])

        # NOTE: an earlier version of this boost appended "consolidated
        # statements of operations"/"consolidated statements of income" to
        # help find the primary statement over the EPS-note variant. It
        # was removed after direct debugging showed it backfiring: that
        # exact phrase appears verbatim in every filing's boilerplate
        # Table of Contents ("Index to Consolidated Financial Statements
        # ... Consolidated Statements of Operations ... 29"), and TOC
        # entries were winning the ranking contest against the actual data
        # rows, which mention "Net income $X" but don't repeat the
        # section-title phrase in the same chunk. The net-income/EPS
        # disambiguation is still handled by the CRITICAL prompt rule in
        # the Synthesizer -- that's a safer place for this distinction
        # than a retrieval-time phrase boost that collides with
        # boilerplate.

        if not expansions:
            return question
        return question + " " + " ".join(expansions)

    def _segment_density_penalty(self, text: str) -> float:
        """Return a score multiplier: 1.0 for chunks that don't look like
        a segment-breakdown table, 0.5 (meaningful downrank, not full
        exclusion -- it might still be the only source available) for
        chunks containing 2+ distinct named-segment-style labels."""
        matches = set(m.group() for m in SEGMENT_LABEL_PATTERN.finditer(text))
        return 0.5 if len(matches) >= 2 else 1.0

    def _looks_like_title_only(self, text: str) -> bool:
        """Detect a chunk that's mostly a page/section title with no real
        financial data -- e.g. "Consolidated statements of income
        JPMorgan Chase & Co./2025 Form 10-K 165" -- a strong signal that
        a table got split across a chunk boundary and the actual numeric
        rows landed in the NEXT chunk instead of this one. Page/form
        numbers and bare years don't count as "real data"; a genuine
        financial figure is either comma-formatted (182,447) or a bare
        4+ digit number that isn't a year."""
        if len(text.strip()) > 250:
            return False
        candidates = re.findall(r'\b\d{1,3}(?:,\d{3})+\b|\b\d{4,}\b', text)
        real_numbers = [n for n in candidates if not re.match(r'^20[0-2]\d$', n)]
        return len(real_numbers) == 0

    def run(self, sub_questions: List[str]) -> Tuple[List[Tuple[Chunk, float, str]], str]:
        """
        Search for each sub-question, deduplicate, return:
        - results: list of (chunk, score, sub_question)
        - context: formatted string for LLM
        """
        sources = self.vector_store.stats()["documents"]
        # Dedup key is (source, chunk_id), not chunk_id alone. Even though
        # VectorStore now guarantees globally-unique chunk_ids, keying dedup
        # on the pair too is cheap defense-in-depth: it means an ID
        # collision across documents (from a stale index, a future
        # refactor, etc.) can never again silently make one company's
        # chunk look like a duplicate of a different company's chunk.
        seen_ids = set()
        all_results = []

        for question in sub_questions:
            search_query = self._expand_query(question)
            try:
                if len(sources) > 1:
                    # Probe first: run one unweighted search to see which
                    # source(s) actually rank well for this specific
                    # sub-question, BEFORE deciding whether to force-split
                    # the retrieval budget evenly across every indexed
                    # document. Without this, a single-company question
                    # (e.g. "Apple's R&D spending") still gets its budget
                    # split 3 ways once 3 documents are indexed, wasting
                    # 2/3 of it on companies the question never mentions
                    # and starving the one source that actually matters.
                    # Genuine cross-company questions naturally produce
                    # top results spread across multiple sources in the
                    # probe, so they still get the balanced treatment.
                    probe_hits = self.vector_store.search(search_query, top_k=self.top_k * 2)
                    relevant_sources = list({chunk.source for chunk, _ in probe_hits})

                    if len(relevant_sources) <= 1:
                        hits = probe_hits[:self.top_k]
                        for chunk, score in hits:
                            key = (chunk.source, chunk.chunk_id)
                            if key not in seen_ids:
                                seen_ids.add(key)
                                all_results.append((chunk, score, question))
                    else:
                        per_source_k = max(2, self.top_k // len(relevant_sources))
                        for src in relevant_sources:
                            hits = self.vector_store.search(search_query, top_k=per_source_k, source=src)
                            for chunk, score in hits:
                                key = (chunk.source, chunk.chunk_id)
                                if key not in seen_ids:
                                    seen_ids.add(key)
                                    all_results.append((chunk, score, question))
                else:
                    hits = self.vector_store.search(search_query, top_k=self.top_k)
                    for chunk, score in hits:
                        key = (chunk.source, chunk.chunk_id)
                        if key not in seen_ids:
                            seen_ids.add(key)
                            all_results.append((chunk, score, question))
            except Exception as e:
                print(f"[Retriever] Search failed for '{question}': {e}")

        # A retrieved chunk that's just a table title with no real numbers
        # (e.g. a table got split across chunk boundaries, header/title in
        # one chunk, data rows in the next) is a dead end on its own --
        # walk forward up to 3 chunks from the same document to find the
        # actual data, stopping as soon as a data-bearing chunk is found.
        # The injected continuation keeps the SAME score as its trigger
        # rather than being discounted: a title-only chunk likely already
        # ranked low to begin with, and further discounting risks the
        # continuation getting cut by the context budget before it ever
        # reaches the Synthesizer.
        augmented = []
        seen_keys = set((c.source, c.chunk_id) for c, _, _ in all_results)
        for chunk, score, q in all_results:
            augmented.append((chunk, score, q))
            if self._looks_like_title_only(chunk.text):
                next_id = chunk.chunk_id + 1
                for _ in range(3):
                    if not (0 <= next_id < len(self.vector_store.chunks)):
                        break
                    next_chunk = self.vector_store.chunks[next_id]
                    if next_chunk.source != chunk.source:
                        break
                    key = (next_chunk.source, next_chunk.chunk_id)
                    if key not in seen_keys:
                        seen_keys.add(key)
                        augmented.append((next_chunk, score, q))
                    if not self._looks_like_title_only(next_chunk.text):
                        break  # found real data -- stop extending
                    next_id += 1
        all_results = augmented

        # Penalize chunks that look like a business-segment breakdown
        # table (see SEGMENT_LABEL_PATTERN) when nothing in this query
        # actually asked about segments -- a confirmed, repeated source
        # of aggregation-level mistakes (citing one segment's total as
        # the firm-wide total). This is a downrank, not exclusion: if
        # it's genuinely the only source retrieved, it still gets used.
        combined_question_text = " ".join(sub_questions)
        if not SEGMENT_QUESTION_PATTERN.search(combined_question_text):
            all_results = [
                (chunk, score * self._segment_density_penalty(chunk.text), q)
                for chunk, score, q in all_results
            ]

        # Sort by score descending (used for per-source ordering below)
        all_results.sort(key=lambda x: x[1], reverse=True)

        # Cap to fit the context budget, round-robinning across sources so
        # a comparison question can't lose one document entirely.
        all_results = self._cap_context_size(all_results)

        # Build formatted context string
        context = self._format_context(all_results)

        return all_results, context

    def _cap_context_size(self, results: List[Tuple[Chunk, float, str]]) -> List[Tuple[Chunk, float, str]]:
        """Round-robin across sources (highest score first within each) until
        the character budget is used up, so every document indexed gets a
        fair share of the final context sent to the LLM."""
        by_source: dict = defaultdict(deque)
        for r in results:
            by_source[r[0].source].append(r)

        capped = []
        total_chars = 0
        sources = list(by_source.keys())
        i = 0
        while sources:
            src = sources[i % len(sources)]
            queue = by_source[src]
            if not queue:
                sources.remove(src)
                if not sources:
                    break
                continue
            chunk, score, question = queue[0]
            chunk_chars = len(chunk.text)
            if total_chars + chunk_chars > MAX_CONTEXT_CHARS and capped:
                break
            queue.popleft()
            capped.append((chunk, score, question))
            total_chars += chunk_chars
            i += 1
        return capped

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
