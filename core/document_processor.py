"""
Document Processor
Handles PDF ingestion, text extraction, and intelligent chunking.
Pure PyMuPDF — no LangChain.
"""

import fitz  # PyMuPDF
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class Chunk:
    text: str
    source: str       # filename
    page: int         # page number
    chunk_id: int     # global index
    char_start: int   # character offset in page


class DocumentProcessor:
    def __init__(self, chunk_size: int = 500, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap

    def load_pdf(self, path: str) -> List[dict]:
        """Extract text from each page of a PDF."""
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

    def _find_sentence_boundary(self, text: str, pos: int) -> int:
        search_window = text[max(0, pos - 150): pos]
        for i in range(len(search_window) - 1, -1, -1):
            if search_window[i] in '.!?' and (
                i + 1 >= len(search_window) or search_window[i + 1] == ' '
            ):
                return max(0, pos - 150) + i + 1
        return pos

    def process(self, pdf_path: str) -> List[Chunk]:
        """Full pipeline: load → clean → chunk."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")
        pages = self.load_pdf(pdf_path)
        chunks = self.chunk_pages(pages, source=path.name)
        return chunks
