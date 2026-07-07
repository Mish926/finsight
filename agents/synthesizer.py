"""
Agent 4: Synthesizer
Takes verified context + original question → structured answer with citations.
Only agent that calls Gemini for generation.
"""


from typing import List, Tuple
import re
from core.document_processor import Chunk
from core.answer_verifier import verify_answer, VerificationIssue, strip_unsupported_claims
from agents.critic import CriticVerdict
from dataclasses import dataclass


SYNTHESIZER_PROMPT = """You are a senior financial analyst. Answer the question below 
using ONLY the provided context. Be precise, factual, and cite your sources.

Question: {question}

Context:
{context}

CRITICAL -- source attribution: Each context block is labeled with its source
filename, e.g. "[Source 3 | apple_10k_2024.pdf | Page 38]". Before using ANY
figure, confirm the source filename actually belongs to the company the
question asks about. NEVER use a number from one company's filing to answer
a question about a different company, even if it seems close or plausible --
this is the single most important rule. If the question asks about Company A
and the only figures you have are from Company B's filing, that means the
answer for Company A is genuinely missing -- say so, do not substitute B's
number or compute a total from B's segments and present it as A's.

CRITICAL -- multi-year tables: Financial statements typically show 2-3 years
side by side (e.g. "Total net sales $391,035 $383,285 $394,328" under a
header naming the years in the same left-to-right order, e.g. "2024 / 2023 /
2022" or "September 28, 2024 / September 30, 2023 / September 24, 2022").
The FIRST number in the row corresponds to the FIRST (usually most recent)
year in the header -- do not assume, always locate the actual header row for
that specific table and match position to position. If no header/year
labels are visible near a number, treat that figure as unlabeled and do not
guess which year it belongs to.

CRITICAL -- which "net income" figure to use: filings with preferred stock
(common for banks) report MULTIPLE net-income-related figures: "Net income"
(the primary, headline figure) and "Net income applicable to common
stockholders" or "...available to common shareholders" (net income minus
preferred dividends -- a smaller number, used specifically for EPS
calculations). When a question just asks for a company's "net income"
without mentioning EPS, shares, or "common stockholders", use the primary
"Net income" line from the main consolidated income statement or financial
summary -- NOT the EPS-note variant, even if it's the more prominent number
in whatever table got retrieved. If you can only find the EPS-note variant,
say so explicitly rather than presenting it as if it were the plain net
income figure.

CRITICAL -- aggregation level: Financial filings contain "Total" figures at
several different levels: firm-wide/consolidated totals, business segment
totals, and geographic/regional totals. A table titled "International
metrics", "By segment", "By region", or similar contains a SUBSET of the
company, not the whole thing -- even though it may literally contain the
words "Total net revenue" or "Total net sales" inside it. When a question
asks for a company's OVERALL total (not a specific segment or region), only
use a figure from a table clearly titled something like "CONSOLIDATED
STATEMENTS OF OPERATIONS/INCOME" or explicitly described as consolidated/
firm-wide. If every figure you can find is a segment or regional subtotal,
say that explicitly rather than presenting a subtotal as the company's
overall total.

MANDATORY SELF-CHECK before you state any headline total figure: privately
verify which table/section heading that number appeared under (e.g. does it
say "International metrics", "By segment", or "CONSOLIDATED STATEMENTS OF
OPERATIONS"?). If the heading names a specific region/segment/division,
that number is NOT the company-wide total -- keep looking for one that is,
or say it's not present if you can't find one. Do this check for EVERY
company in a multi-company comparison, not just the first one.

Do NOT narrate this elimination process in your final answer (e.g. don't
write "Source X says $Y but that's a segment total, and Source Z says $W
but that's also wrong, so..."). Do the checking silently, then present only
the clean final answer with its correct citation. If you genuinely can't
find the consolidated figure after checking, say so directly in one
sentence -- don't walk through every rejected candidate to explain why.

CRITICAL -- do not fabricate lists or categories: for questions asking you
to enumerate or categorize something (e.g. "what are the main risk
factors"), only list items that are ACTUALLY WRITTEN in the context, in
close to their actual wording. Filings frequently reference a list without
including it -- e.g. a sentence like "the other risks and uncertainties
detailed in Part I, Item 1A: Risk Factors" is a POINTER to a list elsewhere
in the document, not the list itself. If that's all you have, say the
context only references the risk factors section without providing the
specific items -- do NOT fill in a plausible-sounding generic list (like
"market risk, credit risk, operational risk...") from general knowledge of
what this type of company usually discloses. A citation next to a
fabricated item is still fabrication.

Instructions:
- First, carefully check whether the answer is actually present in the context --
  including under different wording than the question uses (e.g. "net sales"
  answers a question about "revenue"; "net income" answers "earnings" or "profit").
  Financial tables can look fragmented; read column headers and years carefully
  before concluding a figure is unlabeled or missing.
- If you find the answer: state it directly and confidently. Do NOT prefix it with
  a disclaimer like "the context does not contain..." and then answer anyway --
  either you have the answer or you don't. Pick one.
- Answer directly and concisely
- For every key fact or figure, add a citation like [Source 1, Page 5]
- Use bullet points for multiple data points
- Only if the answer is genuinely absent after checking carefully: say exactly
  what is missing, and do not fabricate or guess a number
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
    def __init__(self, model, vector_store=None):
        self.model = model
        self.vector_store = vector_store
        self.name = "Synthesizer"

    # Phrases that indicate the Synthesizer is declining/hedging rather than
    # confidently answering -- used to derive the displayed confidence
    # badge from what it ACTUALLY produced, instead of the Critic's
    # pre-synthesis guess (which we already know produces false negatives
    # on financial tables -- see the design note above). Without this, a
    # user could see a fully correct, well-cited answer labeled "LOW
    # confidence" just because the Critic misjudged the raw context before
    # the Synthesizer proved it could actually extract the right figure.
    _DECLINE_PHRASES = [
        "not present", "not explicitly stated", "cannot determine",
        "does not contain", "not available", "not provided",
        "cannot accurately", "cannot be determined", "is not stated",
        "is missing", "unable to determine", "context does not",
        "not clearly", "not directly stated"
    ]

    def _infer_confidence(self, answer: str) -> str:
        lower = answer.lower()
        has_citation = bool(re.search(r'\[source\s*\d+', lower))
        declines = any(phrase in lower for phrase in self._DECLINE_PHRASES)
        if has_citation and not declines:
            return "HIGH"
        if has_citation and declines:
            return "MEDIUM"  # partial answer -- some figures found, some not
        return "LOW"

    def _apply_corrections(self, answer: str, issues: List[VerificationIssue]) -> str:
        """Replace a confirmed wrong-year figure with the verified correct
        one, and append a transparent note about what was fixed and why --
        the correction is deterministic (a direct table lookup), so it's
        safe to apply automatically rather than just flagging it."""
        corrected = answer
        notes = []
        for issue in issues:
            digits = issue.cited_value.lstrip('-')
            is_negative = issue.cited_value.startswith('-')
            wrong_variants = [f"-${digits}", f"(${digits})", f"${digits}", digits]
            if is_negative:
                wrong_variants = [f"-${digits}"] + wrong_variants

            correct_clean = re.sub(r'\s+', '', issue.correct_value.replace('$', '')).strip()
            correct_display = f"${correct_clean}"

            replaced = False
            for variant in wrong_variants:
                if variant in corrected:
                    corrected = corrected.replace(variant, correct_display)
                    replaced = True
            if replaced:
                notes.append(
                    f'"{issue.row_label}" was corrected from ${digits} (a different '
                    f'year\'s column) to the verified value {correct_display}.'
                )

        if notes:
            corrected += "\n\n---\n**Verification note:** " + " ".join(notes)
        return corrected

    def run(
        self,
        question: str,
        context: str,
        results: List[Tuple[Chunk, float, str]],
        critic_verdict: CriticVerdict,
        sub_questions: List[str]
    ) -> SynthesisResult:
        """Generate final answer with citations."""

        # Design note: synthesis is intentionally NOT hard-gated on
        # critic_verdict.sufficient. The Critic runs on the same small
        # (Llama 3.1 8B) model as everything else in this pipeline, and in
        # practice produces false negatives on financial tables -- it can
        # see the exact figure in context and still call it "insufficient"
        # because the surrounding table formatting looks ambiguous to it.
        # Hard-gating on that would silently throw away correct answers.
        # Instead, the Synthesizer always attempts an answer from whatever
        # context it has, and the Critic's confidence/verdict is surfaced
        # to the user as advisory context (the confidence badge in the UI)
        # rather than a block. The Synthesizer's own instructions (below)
        # are responsible for declining when the answer genuinely isn't
        # there.

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

        # Deterministic verification pass: check every cited figure that
        # points to a clean table chunk against the actual column for the
        # year the question asked about. This catches the specific,
        # high-stakes failure mode where the model cites a real number
        # from a real source but reads the wrong year's column (e.g.
        # stating a company's 2022 loss as its 2024 net income) -- a
        # prompt instruction reduces this, but can't guarantee it the way
        # a direct table lookup can.
        all_chunks = self.vector_store.chunks if self.vector_store is not None else None
        issues = verify_answer(answer, results, question, all_chunks=all_chunks)
        if issues:
            answer = self._apply_corrections(answer, issues)

        # Separate check: a derived comparison in prose (e.g. "X times
        # higher") can be nonsensical even when every individual cited
        # figure is correct -- this isn't a citation problem, so the
        # check above can't catch it. Verify any stated ratio/percent
        # claim against the answer's own cited figures and strip it if
        # unsupported, rather than leave a fabricated comparison standing.
        answer = strip_unsupported_claims(answer)

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
            confidence=self._infer_confidence(answer),
            sub_questions=sub_questions,
            verdict=critic_verdict.verdict
        )
