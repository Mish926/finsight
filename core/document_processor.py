"""
Document Processor
Handles PDF ingestion, text extraction, table extraction, and chunking.
"""

import fitz  # PyMuPDF -- flowing prose text extraction
import pdfplumber  # structure-aware table extraction
import gc
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    text: str
    source: str       # filename
    page: int         # page number
    chunk_id: int     # global index (reassigned uniquely by VectorStore.add_chunks)
    char_start: int   # character offset in page
    is_table: bool = False  # True for structured-table chunks (see below)


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_pdf(self, path: str) -> List[dict]:
        """Extract flowing prose text from each page of a PDF."""
        doc = fitz.open(path)
        pages = []
        for page_num, page in enumerate(doc, start=1):
            text = page.get_text("text")
            text = self._clean(text)
            if text.strip():
                pages.append({"page": page_num, "text": text})
        doc.close()
        return pages

    def _clean(self, text: str) -> str:
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'[ \t]{2,}', ' ', text)
        text = re.sub(r'\x00', '', text)
        return text.strip()

    def load_tables(self, path: str) -> List[dict]:
        """Extract tables with pdfplumber, preserving row/column structure.

        PyMuPDF's plain-text extraction reads a page as one linear stream,
        which works fine for prose but scrambles wide, multi-column
        financial tables -- e.g. a table with segment columns (Corporate)
        interleaved with firm-wide Total columns across three years gets
        linearized as "2024 2023 2022 2024 2023 2022 Corporate Corporate
        Corporate Total Total Total Total net revenue 17,394 8,038 80
        180,593 162,366 132,277", which is nearly impossible to correctly
        map back to (which year, which segment) even for a careful reader.
        pdfplumber detects the actual table grid and returns it as proper
        rows/columns, which we then render as clean markdown -- so a figure
        and its year/segment header stay unambiguously connected.

        Two detection strategies are tried: pdfplumber's default "lines"
        strategy (fast, accurate when the table has visible ruling/grid
        lines) and a "text" strategy based on whitespace alignment (needed
        because most real financial-report tables -- including JPMorgan's
        segment tables -- are NOT drawn with visible borders at all; they're
        just aligned columns of text, which "lines" silently finds zero
        tables for). Text-strategy runs only as a fallback when line-based
        detection finds nothing, since it's more prone to noisy blank rows.

        MEMORY: pdfplumber's Page objects cache their parsed layout data
        (characters, lines, rects) the first time it's accessed, and that
        cache is NOT released automatically as you iterate through many
        pages. For a large filing (JPMorgan's 10-K runs 400+ pages), that
        accumulates across the whole document and can be a significant
        contributor to out-of-memory crashes on a constrained host. Each
        page's cache is explicitly flushed after processing it, which
        pdfplumber provides specifically for this use case.
        """
        tables_by_page = []
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                try:
                    tables = page.extract_tables()
                    if not tables:
                        tables = page.extract_tables(table_settings={
                            "vertical_strategy": "text",
                            "horizontal_strategy": "text",
                        })
                except Exception as e:
                    print(f"[DocumentProcessor] Table extraction failed on page {page_num}: {e}")
                    page.flush_cache()
                    continue
                for table in tables:
                    table = self._drop_blank_rows(table)
                    if not table or len(table) < 2:
                        continue
                    non_empty = sum(1 for row in table for cell in row if cell and cell.strip())
                    total_cells = sum(len(row) for row in table)
                    if total_cells == 0 or non_empty / total_cells < 0.3:
                        continue
                    if self._header_looks_misaligned(table):
                        # Text-strategy column detection can fail on very
                        # wide/dense tables (e.g. a bank's 4-segments x
                        # 3-years = 12-column summary table), producing a
                        # table that LOOKS clean and structured but has
                        # cells grouped into the wrong columns. That's worse
                        # than the garbled prose fallback, because it reads
                        # as authoritative while still being wrong. Skip it
                        # -- the prose chunk for this page still exists and
                        # is a more honest (if harder to read) source.
                        print(f"[DocumentProcessor] Skipping likely-misaligned table on page {page_num}")
                        continue
                    md = self._table_to_markdown(table)
                    if len(md.strip()) < 30:
                        continue
                    tables_by_page.append({"page": page_num, "text": md})
                page.flush_cache()
                if page_num % 50 == 0:
                    # CPython doesn't always return freed memory to the OS
                    # promptly within a long-running loop -- an explicit
                    # nudge every 50 pages helps keep peak memory down on
                    # large (400+ page) filings.
                    gc.collect()
        return tables_by_page

    def _header_looks_misaligned(self, table: List[List], header_rows: int = 2) -> bool:
        """Heuristic: a legitimate wide financial table can validly repeat a
        group label several times in a row (e.g. "Corporate Corporate
        Corporate Total Total Total" for 3 years x 2 groups) -- that's a
        normal CONTIGUOUS block and not a problem. What signals genuine
        misalignment is the same label appearing 3+ times SCATTERED with
        other different labels interspersed between occurrences (e.g.
        "Consumer... Commercial... Asset... Consumer... Consumer..." --
        segment names that should be neighbors got separated), which is
        what text-strategy clustering does when it can't cleanly resolve a
        very wide/dense column grid."""
        from collections import defaultdict
        positions = defaultdict(list)
        idx = 0
        for row in table[:header_rows]:
            for cell in row:
                if cell and cell.strip():
                    text = cell.strip()
                    if not text.replace(',', '').replace('.', '').replace('-', '').replace('$', '').isdigit():
                        stripped = re.sub(r'\s*\d{4}\s*$', '', text).strip()
                        if stripped:
                            positions[stripped].append(idx)
                idx += 1
        for label, pos_list in positions.items():
            if len(pos_list) >= 3:
                span = max(pos_list) - min(pos_list) + 1
                if span > len(pos_list):
                    return True  # gaps between occurrences -> scattered, not a clean group
        return False

    def _drop_blank_rows(self, table: List[List]) -> List[List]:
        """Text-strategy detection sometimes emits rows that are entirely
        empty cells (whitespace-clustering artifacts) -- drop those."""
        return [row for row in table if any(cell and cell.strip() for cell in row)]

    def _table_to_markdown(self, table: List[List]) -> str:
        """Render a pdfplumber table (list of row lists) as markdown, which
        keeps every cell's row/column position -- and therefore its year/
        segment meaning -- explicit and unambiguous for the LLM."""
        rows = [[(cell.strip() if cell else "") for cell in row] for row in table]
        # Normalize ragged rows to the same width
        width = max(len(r) for r in rows)
        rows = [r + [""] * (width - len(r)) for r in rows]

        lines = []
        header = rows[0]
        lines.append("| " + " | ".join(c if c else " " for c in header) + " |")
        lines.append("| " + " | ".join(["---"] * width) + " |")
        for row in rows[1:]:
            lines.append("| " + " | ".join(c if c else " " for c in row) + " |")
        return "\n".join(lines)

    def chunk_pages(self, pages: List[dict], source: str) -> List[Chunk]:
        """Split pages into overlapping chunks at sentence boundaries."""
        chunks = []
        chunk_id = 0

        for page_data in pages:
            page_num = page_data["page"]
            text = page_data["text"]

            start = 0
            while start < len(text):
                end = start + self.chunk_size

                if end < len(text):
                    boundary = self._find_sentence_boundary(text, end)
                    end = boundary if boundary > start else end

                chunk_text = text[start:end].strip()

                if len(chunk_text) > 50:
                    chunks.append(Chunk(
                        text=chunk_text,
                        source=source,
                        page=page_num,
                        chunk_id=chunk_id,
                        char_start=start
                    ))
                    chunk_id += 1

                start = end - self.overlap
                if start >= len(text):
                    break

        return chunks

    def chunk_tables(self, tables: List[dict], source: str, start_id: int = 0) -> List[Chunk]:
        """Each extracted table becomes its own chunk -- tables are already
        a self-contained, structured unit; splitting one by character count
        the way prose is chunked would just reintroduce the header/data
        separation problem we're trying to avoid."""
        chunks = []
        chunk_id = start_id
        for t in tables:
            chunks.append(Chunk(
                text=t["text"],
                source=source,
                page=t["page"],
                chunk_id=chunk_id,
                char_start=0,
                is_table=True
            ))
            chunk_id += 1
        return chunks

    def _find_sentence_boundary(self, text: str, pos: int) -> int:
        search_window = text[max(0, pos - 150): pos]
        for i in range(len(search_window) - 1, -1, -1):
            if search_window[i] in '.!?' and (
                i + 1 >= len(search_window) or search_window[i + 1] == ' '
            ):
                return max(0, pos - 150) + i + 1
        return pos

    def process(self, pdf_path: str) -> List[Chunk]:
        """Full pipeline: prose chunks + structured table chunks, combined.
        chunk_id here is only locally unique per document -- VectorStore
        reassigns globally-unique ids across the whole index when chunks
        are added, so upload order/count never causes collisions."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        pages = self.load_pdf(pdf_path)
        prose_chunks = self.chunk_pages(pages, source=path.name)

        tables = self.load_tables(pdf_path)
        table_chunks = self.chunk_tables(tables, source=path.name, start_id=len(prose_chunks))

        print(f"  Extracted {len(prose_chunks)} prose chunks + {len(table_chunks)} table chunks")
        return prose_chunks + table_chunks
