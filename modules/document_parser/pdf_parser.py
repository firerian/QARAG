import os
import re
from typing import List, Optional

from modules.document_parser.base import (
    BaseDocumentParser,
    Chunk,
    ChunkingConfig,
    ProcessResult,
)
from modules.document_parser.chunking import ChunkingStrategy, create_chunking_strategy
from modules.logger import get_logger

logger = get_logger(__name__)

PDF_SUPPORT = False
try:
    import pdfplumber
    PDF_SUPPORT = True
except ImportError:
    try:
        import PyPDF2
        PDF_SUPPORT = True
        pdfplumber = None
    except ImportError:
        pass


class PDFDocumentParser(BaseDocumentParser):
    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        remove_headers_footers: bool = True,
    ):
        super().__init__(chunking_config)
        self.supported_extensions = [".pdf"]
        self.remove_headers_footers = remove_headers_footers
        self.chunking_strategy = create_chunking_strategy(
            "pdf", self.chunking_config
        )

    def parse(self, file_path: str) -> ProcessResult:
        if not PDF_SUPPORT:
            return ProcessResult(
                success=False,
                errors=["PDF support requires 'pdfplumber' or 'PyPDF2' package. Install with: pip install pdfplumber"],
            )

        if not os.path.exists(file_path):
            return ProcessResult(
                success=False,
                errors=[f"File not found: {file_path}"],
            )

        all_chunks: List[Chunk] = []

        try:
            extracted_pages = self._extract_pages(file_path)
        except Exception as e:
            logger.error("Failed to extract text from PDF %s: %s", file_path, e)
            return ProcessResult(
                success=False,
                errors=[f"PDF extraction failed: {str(e)}"],
            )

        if not extracted_pages:
            return ProcessResult(
                success=False,
                errors=["No extractable text found in PDF"],
            )

        total_pages = len(extracted_pages)
        for page_num, page_text in extracted_pages:
            if not page_text or not page_text.strip():
                continue

            page_metadata = {
                "file_name": os.path.basename(file_path),
                "file_path": os.path.abspath(file_path),
                "file_type": "pdf",
                "page_number": page_num,
                "total_pages": total_pages,
            }

            page_chunks = self.chunking_strategy.chunk_text(
                page_text, metadata=page_metadata
            )
            all_chunks.extend(page_chunks)

        if not all_chunks:
            return ProcessResult(
                success=False,
                errors=["No chunks generated from PDF content"],
            )

        return ProcessResult(
            chunks=all_chunks,
            metadata={
                "file_name": os.path.basename(file_path),
                "file_type": "pdf",
                "total_pages": total_pages,
                "total_chunks": len(all_chunks),
            },
        )

    def extract_text(self, file_path: str) -> str:
        if not PDF_SUPPORT:
            raise ImportError(
                "PDF support requires 'pdfplumber' or 'PyPDF2'. Install with: pip install pdfplumber"
            )

        pages = self._extract_pages(file_path)
        return "\n\n".join(text for _, text in pages)

    def _extract_pages(self, file_path: str) -> List[tuple]:
        if pdfplumber is not None:
            return self._extract_with_pdfplumber(file_path)
        else:
            return self._extract_with_pypdf2(file_path)

    def _extract_with_pdfplumber(self, file_path: str) -> List[tuple]:
        pages: List[tuple] = []
        with pdfplumber.open(file_path) as pdf:
            for i, page in enumerate(pdf.pages, 1):
                try:
                    text = page.extract_text()
                    if text and self.remove_headers_footers:
                        text = self._clean_page_text(text)
                    pages.append((i, text or ""))
                except Exception as e:
                    logger.warning("Failed to extract page %d from %s: %s", i, file_path, e)
                    pages.append((i, ""))
        return pages

    def _extract_with_pypdf2(self, file_path: str) -> List[tuple]:
        pages: List[tuple] = []
        with open(file_path, "rb") as f:
            reader = PyPDF2.PdfReader(f)
            for i, page in enumerate(reader.pages, 1):
                try:
                    text = page.extract_text()
                    if text and self.remove_headers_footers:
                        text = self._clean_page_text(text)
                    pages.append((i, text or ""))
                except Exception as e:
                    logger.warning("Failed to extract page %d from %s: %s", i, file_path, e)
                    pages.append((i, ""))
        return pages

    def _clean_page_text(self, text: str) -> str:
        lines = text.splitlines()
        if not lines:
            return text

        if len(lines) >= 3:
            first_line = lines[0].strip()
            if self._looks_like_header_footer(first_line):
                lines = lines[1:]

        if len(lines) >= 3:
            last_line = lines[-1].strip()
            if self._looks_like_header_footer(last_line):
                lines = lines[:-1]

        page_number_pattern = re.compile(r'^\s*\d{1,4}\s*$')
        if lines and page_number_pattern.match(lines[0].strip()):
            lines = lines[1:]
        if lines and page_number_pattern.match(lines[-1].strip()):
            lines = lines[:-1]

        return "\n".join(lines)

    def _looks_like_header_footer(self, line: str) -> bool:
        if not line:
            return False

        if len(line) < 3:
            return True

        url_pattern = re.compile(r'https?://|www\.')
        if url_pattern.search(line):
            return True

        date_pattern = re.compile(r'^\d{4}[-/]\d{2}[-/]\d{2}')
        if date_pattern.match(line):
            return True

        chinese_page = re.compile(r'^第\s*\d+\s*页')
        if chinese_page.match(line):
            return True

        english_page = re.compile(r'^Page\s+\d+', re.IGNORECASE)
        if english_page.match(line):
            return True

        confidential = re.compile(r'^Confidential|^Draft|^Internal', re.IGNORECASE)
        if confidential.match(line):
            return True

        return False
