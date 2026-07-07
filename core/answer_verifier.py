"""
Answer Verifier — catches a specific, high-stakes RAG failure mode: the
Synthesizer citing a real number from a real source, but from the WRONG
YEAR'S column in a multi-year table (e.g. stating Amazon's 2022 net loss
of $(2,722)M as if it were the 2024 figure of $59,248M).

This is deterministic verification, not another prompt instruction: for
each cited figure, if its source is a clean markdown table chunk (built by
DocumentProcessor's pdfplumber extraction), the table's header row and data
rows are parsed directly, and the cited number's column position is checked
against the column that actually corresponds to the year the question
asked about. A mismatch means the wrong column was read — deterministically,
not probabilistically -- and the correct value can be read directly from
the same row.

Only table chunks (chunk.is_table=True) can be verified this way, since
they have a clean, structurally reliable header. Prose chunks lack that
guarantee, so citations from prose fall through as unverifiable (not
flagged as wrong -- just not checked).
"""

import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from core.document_processor import Chunk


CITATION_NUMBER_PATTERN = re.compile(
    r'(-)?\$\s*\(?([\d,]+(?:\.\d+)?)\)?\s*(?:million|billion)?[^\[\]]{0,150}?\[Source\s*(\d+)',
    re.DOTALL
)


@dataclass
class VerificationIssue:
    source_num: int
    cited_value: str
    correct_value: str
    row_label: str


def extract_target_year(question: str) -> Optional[str]:
    match = re.search(r'\b(20[12]\d)\b', question)
    return match.group(1) if match else None


def normalize_number(cell: str) -> Optional[str]:
    """Extract a normalized numeric string from a table cell, preserving
    sign (parenthesized numbers, e.g. "(2,722)", are negative in financial
    statements -- but a plain leading minus, e.g. "-2,722", also means
    negative and must be detected too)."""
    if not cell:
        return None
    s = cell.replace(',', '').replace('$', '').strip()
    negative = (s.startswith('(') and s.endswith(')')) or s.startswith('-')
    s = s.strip('()').strip()
    if s.startswith('-'):
        s = s[1:].strip()
    match = re.search(r'\d+(?:\.\d+)?', s)
    if not match:
        return None
    val = match.group()
    return ('-' + val) if negative else val


def parse_markdown_table(table_text: str) -> Tuple[Optional[List[str]], Optional[List[List[str]]]]:
    """Parse a DocumentProcessor-generated markdown table chunk into
    (header_cells, data_rows), skipping the '| --- |' separator row."""
    lines = [l.strip() for l in table_text.strip().split('\n') if l.strip().startswith('|')]
    rows = [[c.strip() for c in line.strip('|').split('|')] for line in lines]
    if len(rows) < 2:
        return None, None
    header = rows[0]
    data_rows = [r for r in rows[1:] if not all(set(c) <= set('- ') for c in r if c)]
    return header, data_rows


def find_year_column(header: List[str], target_year: str) -> Optional[int]:
    for i, cell in enumerate(header):
        if target_year in cell:
            return i
    return None


def find_year_clusters(text: str, max_gap: int = 15) -> List[Tuple[List[str], int, int]]:
    """Find clusters of 2-4 year-like tokens (19xx/20xx) that appear close
    together in the text -- a likely table header row in linearized prose,
    where each year ended up on its own line/token during PDF text
    extraction. Returns (years_in_order, cluster_start, cluster_end) for
    each cluster found."""
    year_matches = list(re.finditer(r'\b(19|20)\d{2}\b', text))
    clusters = []
    i = 0
    while i < len(year_matches):
        cluster = [year_matches[i]]
        j = i + 1
        while j < len(year_matches) and year_matches[j].start() - cluster[-1].end() < max_gap:
            cluster.append(year_matches[j])
            j += 1
        if len(cluster) >= 2:
            years = [m.group() for m in cluster]
            clusters.append((years, cluster[0].start(), cluster[-1].end()))
        i = j if j > i else i + 1
    return clusters


NUMBER_TOKEN_PATTERN = re.compile(r'\(?-?\$?\s?[\d,]{2,}(?:\.\d+)?\)?')


def find_number_clusters(text: str, max_gap: int = 20) -> List[Tuple[List[str], int, int]]:
    """Same clustering idea as find_year_clusters, but for numeric data
    rows -- financial figures in linearized PDF text tend to land close
    together (separated only by '$' signs and whitespace) even when the
    row's text label is on a separate line above them."""
    num_matches = list(NUMBER_TOKEN_PATTERN.finditer(text))
    clusters = []
    i = 0
    while i < len(num_matches):
        cluster = [num_matches[i]]
        j = i + 1
        while j < len(num_matches) and num_matches[j].start() - cluster[-1].end() < max_gap:
            cluster.append(num_matches[j])
            j += 1
        if len(cluster) >= 2:
            vals = [m.group() for m in cluster]
            clusters.append((vals, cluster[0].start(), cluster[-1].end()))
        i = j if j > i else i + 1
    return clusters


def abs_value(normalized: Optional[str]) -> Optional[str]:
    """Strip sign for matching purposes. The Synthesizer's prose doesn't
    always textually preserve a negative sign even when the underlying
    figure is negative -- e.g. "a loss of $2,722 million" has no literal
    minus sign, even though the source table shows "(2,722)". Requiring
    an exact signed match would silently fail to even locate that number
    in the table at all. Column-position verification only needs to find
    WHICH cell the cited magnitude came from; the correction value pulled
    from the table already carries its own correct sign regardless."""
    if normalized is None:
        return None
    return normalized.lstrip('-')


def verify_figure_in_prose(chunk_text: str, cited_value: str, target_year: str) -> Optional[VerificationIssue]:
    """Fallback verification for chunks that AREN'T clean pdfplumber
    tables (is_table=False) -- e.g. when pdfplumber's table detection
    didn't recognize a given table at all, and it only exists in the
    corpus as PyMuPDF's linearized text. This is a looser heuristic than
    the markdown-table check (proximity clustering instead of an explicit
    grid), so it's tried only as a fallback, not the primary check."""
    cited_norm = normalize_number(cited_value)
    if cited_norm is None:
        return None
    cited_abs = abs_value(cited_norm)

    year_clusters = find_year_clusters(chunk_text)
    number_clusters = find_number_clusters(chunk_text)

    for vals, start, end in number_clusters:
        norms = [normalize_number(v) for v in vals]
        abs_norms = [abs_value(n) for n in norms]
        if cited_abs not in abs_norms:
            continue
        cited_col = abs_norms.index(cited_abs)

        candidate_headers = [
            (years, h_start, h_end) for years, h_start, h_end in year_clusters
            if h_end <= start and len(years) == len(vals)
        ]
        if not candidate_headers:
            continue
        years, h_start, h_end = max(candidate_headers, key=lambda c: c[2])

        if target_year not in years:
            continue
        year_col = years.index(target_year)

        if cited_col == year_col:
            return None  # verified OK
        return VerificationIssue(
            source_num=0,
            cited_value=cited_value,
            correct_value=vals[year_col].strip(),
            row_label="this figure"
        )
    return None


def verify_figure(chunk: Chunk, cited_value: str, target_year: str, extra_context: str = "") -> Optional[VerificationIssue]:
    """Returns a VerificationIssue if a mismatch is found, None if the
    figure checks out OR can't be verified (not a table / no year header /
    number not found in the table -- all treated as "don't know", not
    "wrong", since we'd rather stay silent than raise a false alarm)."""
    if getattr(chunk, "is_table", False):
        header, data_rows = parse_markdown_table(chunk.text)
        if header and data_rows:
            year_col = find_year_column(header, target_year)
            cited_norm = normalize_number(cited_value)
            cited_abs = abs_value(cited_norm)
            if year_col is not None and cited_abs is not None:
                for row in data_rows:
                    row_norms = [normalize_number(c) for c in row]
                    abs_row_norms = [abs_value(n) for n in row_norms]
                    if cited_abs in abs_row_norms:
                        cited_col = abs_row_norms.index(cited_abs)
                        if cited_col == year_col:
                            return None  # verified OK
                        if year_col < len(row) and row[year_col].strip():
                            return VerificationIssue(
                                source_num=0,
                                cited_value=cited_value,
                                correct_value=row[year_col].strip(),
                                row_label=row[0].strip() if row[0] else "this figure"
                            )
                        return None
        return None  # table parse failed -- don't fall through to the
        # looser prose heuristic on something we know IS a table; better
        # to stay silent than risk a bad match on malformed table text

    # Not a table chunk -- try the looser prose-proximity heuristic instead.
    # If the chunk alone has no year header (common for long tables that
    # got split across chunk boundaries -- the header can end up in a
    # separate, earlier chunk than the data row), fall back to searching
    # across nearby same-source chunks too.
    issue = verify_figure_in_prose(chunk.text, cited_value, target_year)
    if issue is not None or not extra_context:
        return issue
    return verify_figure_in_prose(extra_context + "\n" + chunk.text, cited_value, target_year)


def gather_nearby_context(chunk: Chunk, results: List[Tuple[Chunk, float, str]],
                           all_chunks: Optional[List[Chunk]] = None, page_window: int = 2) -> str:
    """Concatenate text from other chunks that share the same source and
    are within page_window pages of the cited chunk. Long tables (e.g. a
    full multi-line-item income statement) can exceed a single chunk's
    size and get split -- the year header row ends up in one chunk, a data
    row like "Net income" in another. Searching this combined window
    recovers the header/data pairing that a single-chunk check would miss.

    Prefers searching the FULL indexed corpus (all_chunks) over just this
    query's retrieved subset (results) when available: the header chunk is
    frequently not part of what got retrieved for a given question at all
    (it's a different sub-question's territory), so limiting the search to
    `results` misses it even when it exists in the corpus."""
    pool = all_chunks if all_chunks is not None else [c for c, _, _ in results]
    parts = []
    for other_chunk in pool:
        if other_chunk.source == chunk.source and other_chunk is not chunk \
                and abs(other_chunk.page - chunk.page) <= page_window:
            parts.append(other_chunk.text)
    return "\n".join(parts)


def verify_answer(answer: str, results: List[Tuple[Chunk, float, str]], question: str,
                   all_chunks: Optional[List[Chunk]] = None) -> List[VerificationIssue]:
    """Scan an answer for cited figures and check each one that points to
    a table chunk against the year the question asked about. Returns a
    list of confirmed mismatches (empty list if everything checks out or
    nothing was verifiable). Pass all_chunks (the full indexed corpus) to
    let verification recover year headers that landed outside this
    query's retrieved set entirely."""
    target_year = extract_target_year(question)
    if not target_year:
        return []

    issues = []
    for match in CITATION_NUMBER_PATTERN.finditer(answer):
        sign, digits, source_num_str = match.group(1), match.group(2), match.group(3)
        cited_value = (sign or '') + digits
        source_idx = int(source_num_str) - 1
        if source_idx < 0 or source_idx >= len(results):
            continue
        chunk = results[source_idx][0]
        extra_context = gather_nearby_context(chunk, results, all_chunks=all_chunks)
        issue = verify_figure(chunk, cited_value, target_year, extra_context)
        if issue:
            issue.source_num = int(source_num_str)
            issues.append(issue)
    return issues


# --- Ratio/percent claim sanity-checking ---
#
# A separate failure mode from wrong-year-column: the Synthesizer can cite
# individually correct figures but then state a nonsensical DERIVED
# comparison in prose -- e.g. "Apple's net income was more than 4,000
# times higher than JPMorgan's" when the actual ratio between the two
# cited numbers is ~1.6x. This isn't a citation problem (no single number
# is wrong), so the year-column verifier above can't catch it -- it needs
# its own check: does ANY pair of the actual cited figures in the answer
# support the claimed ratio? If not, the claim is unsupported and gets
# stripped rather than left standing (or "corrected" to a guessed value,
# which risks inserting a different wrong number).

RATIO_CLAIM_PATTERN = re.compile(
    r'(\d[\d,]*(?:\.\d+)?)\s*(?:times|x)\s+(higher|more|greater|larger|bigger|lower|less|smaller)',
    re.IGNORECASE
)
PERCENT_CLAIM_PATTERN = re.compile(
    r'(\d[\d,]*(?:\.\d+)?)\s*%\s+(higher|more|greater|larger|bigger|lower|less|smaller)',
    re.IGNORECASE
)
RATIO_SUPPORT_TOLERANCE = 0.3  # allow up to 30% relative error before flagging as unsupported


def extract_cited_values(answer: str) -> List[float]:
    """All signed numeric values cited anywhere in the answer (regardless
    of which source), used as the pool of "actual" figures a comparative
    claim could legitimately be based on."""
    values = []
    for m in CITATION_NUMBER_PATTERN.finditer(answer):
        sign, digits = m.group(1), m.group(2)
        try:
            val = float(digits.replace(',', ''))
        except ValueError:
            continue
        values.append(-val if sign else val)
    return values


def find_unsupported_ratio_claims(answer: str) -> List[str]:
    """Return the exact matched text (e.g. "4,000 times higher") of any
    comparative claim that no pair of the answer's own cited figures can
    support, even loosely."""
    values = extract_cited_values(answer)
    unsupported = []

    for pattern, is_percent in [(RATIO_CLAIM_PATTERN, False), (PERCENT_CLAIM_PATTERN, True)]:
        for m in pattern.finditer(answer):
            try:
                stated = float(m.group(1).replace(',', ''))
            except ValueError:
                continue

            best_rel_error = None
            for i in range(len(values)):
                for j in range(i + 1, len(values)):
                    a, b = values[i], values[j]
                    if a == 0 or b == 0:
                        continue
                    bigger, smaller = max(abs(a), abs(b)), min(abs(a), abs(b))
                    actual = ((bigger - smaller) / smaller * 100) if is_percent else (bigger / smaller)
                    if actual == 0:
                        continue
                    rel_error = abs(stated - actual) / actual
                    if best_rel_error is None or rel_error < best_rel_error:
                        best_rel_error = rel_error

            if best_rel_error is None or best_rel_error > RATIO_SUPPORT_TOLERANCE:
                unsupported.append(m.group(0))

    return unsupported


def strip_unsupported_claims(answer: str) -> str:
    """Remove sentences containing an unsupported ratio/percent claim
    entirely, rather than leave a fabricated comparison in the answer or
    guess at a replacement number. Appends a transparency note listing
    what was removed and why."""
    unsupported = find_unsupported_ratio_claims(answer)
    if not unsupported:
        return answer

    cleaned = answer
    removed_notes = []
    for claim_text in unsupported:
        idx = cleaned.find(claim_text)
        if idx == -1:
            continue
        s_start, s_end = _find_clause_bounds(cleaned, idx, len(claim_text))
        cleaned = cleaned[:s_start] + cleaned[s_end:]
        removed_notes.append(f'A claim ("{claim_text}") was removed because no combination of the '
                              f'figures actually cited in this answer supports it.')

    if removed_notes:
        cleaned = cleaned.rstrip() + "\n\n---\n**Verification note:** " + " ".join(removed_notes)
    return cleaned


def _find_clause_bounds(text: str, pos: int, match_len: int, max_lookback: int = 200, max_lookahead: int = 200):
    """Find a safely-bounded span around `pos` to remove -- using periods
    OR newlines/bullet breaks as boundaries (bulleted answers often have
    no periods at all), and never searching further back/forward than
    max_lookback/max_lookahead so a missing boundary can't cause the
    whole answer to be deleted."""
    search_start = max(0, pos - max_lookback)
    window_before = text[search_start:pos]
    boundary_positions = [m.end() for m in re.finditer(r'[.\n]', window_before)]
    s_start = search_start + boundary_positions[-1] if boundary_positions else search_start

    end_pos = pos + match_len
    search_end = min(len(text), end_pos + max_lookahead)
    window_after = text[end_pos:search_end]
    end_match = re.search(r'[.\n]', window_after)
    s_end = end_pos + end_match.end() if end_match else search_end

    return s_start, s_end
