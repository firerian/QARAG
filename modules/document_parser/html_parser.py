import os
import re
from typing import List, Optional, Set

from modules.document_parser.base import (
    BaseDocumentParser,
    Chunk,
    ChunkingConfig,
    ProcessResult,
)
from modules.document_parser.chunking import ChunkingStrategy, create_chunking_strategy
from modules.logger import get_logger

logger = get_logger(__name__)

HTML_SUPPORT = False
try:
    from bs4 import BeautifulSoup, Tag, NavigableString
    HTML_SUPPORT = True
except ImportError:
    pass

SKIP_TAGS: Set[str] = {
    "script", "style", "nav", "header", "footer",
    "aside", "noscript", "iframe", "object", "embed",
    "svg", "canvas", "form", "input", "button",
}

SKIP_CLASS_PATTERNS = [
    re.compile(r'nav', re.IGNORECASE),
    re.compile(r'menu', re.IGNORECASE),
    re.compile(r'sidebar', re.IGNORECASE),
    re.compile(r'footer', re.IGNORECASE),
    re.compile(r'advertisement|ad-|ads-|banner', re.IGNORECASE),
    re.compile(r'social|share', re.IGNORECASE),
    re.compile(r'comment', re.IGNORECASE),
    re.compile(r'cookie|popup|modal', re.IGNORECASE),
]

SEMANTIC_TAGS: Set[str] = {
    "article", "main", "section", "div", "p",
    "h1", "h2", "h3", "h4", "h5", "h6",
    "blockquote", "pre", "code", "ul", "ol", "li",
    "table", "dl", "dt", "dd", "figure", "figcaption",
}


class HTMLDocumentParser(BaseDocumentParser):
    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
        extract_links: bool = False,
        clean_whitespace: bool = True,
    ):
        super().__init__(chunking_config)
        self.supported_extensions = [".html", ".htm"]
        self.extract_links = extract_links
        self.clean_whitespace = clean_whitespace
        self.chunking_strategy = create_chunking_strategy(
            "html", self.chunking_config
        )

    def parse(self, file_path: str) -> ProcessResult:
        if not HTML_SUPPORT:
            return ProcessResult(
                success=False,
                errors=["HTML support requires 'beautifulsoup4' package. Install with: pip install beautifulsoup4 lxml"],
            )

        if not os.path.exists(file_path):
            return ProcessResult(
                success=False,
                errors=[f"File not found: {file_path}"],
            )

        try:
            text = self.extract_text(file_path)
        except Exception as e:
            logger.error("Failed to extract text from HTML %s: %s", file_path, e)
            return ProcessResult(
                success=False,
                errors=[f"HTML extraction failed: {str(e)}"],
            )

        if not text or not text.strip():
            return ProcessResult(
                success=False,
                errors=["No extractable content found in HTML"],
            )

        base_metadata = {
            "file_name": os.path.basename(file_path),
            "file_path": os.path.abspath(file_path),
            "file_type": "html",
        }

        all_chunks = self.chunking_strategy.chunk_text(
            text, metadata=base_metadata
        )

        if not all_chunks:
            return ProcessResult(
                success=False,
                errors=["No chunks generated from HTML content"],
            )

        return ProcessResult(
            chunks=all_chunks,
            metadata={
                "file_name": os.path.basename(file_path),
                "file_type": "html",
                "total_chunks": len(all_chunks),
            },
        )

    def extract_text(self, file_path: str) -> str:
        if not HTML_SUPPORT:
            raise ImportError(
                "HTML support requires 'beautifulsoup4'. Install with: pip install beautifulsoup4 lxml"
            )

        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        try:
            soup = BeautifulSoup(content, "lxml")
        except Exception:
            soup = BeautifulSoup(content, "html.parser")

        for tag_name in SKIP_TAGS:
            for tag in soup.find_all(tag_name):
                tag.decompose()

        for element in soup.find_all(True):
            if self._should_skip_element(element):
                element.decompose()

        body = soup.find("body")
        if body is None:
            body = soup

        extracted = self._extract_from_element(body)
        extracted = extracted.strip()

        if self.clean_whitespace:
            extracted = re.sub(r'[ \t]+', ' ', extracted)
            extracted = re.sub(r'\n{3,}', '\n\n', extracted)

        return extracted

    def _should_skip_element(self, element: "Tag") -> bool:
        if not hasattr(element, 'get'):
            return False

        class_attr = element.get('class', [])
        id_attr = element.get('id', '')

        if not class_attr and not id_attr:
            return False

        combined = ' '.join(
            (class_attr if isinstance(class_attr, list) else [class_attr])
            + [id_attr]
        )

        for pattern in SKIP_CLASS_PATTERNS:
            if pattern.search(combined):
                return True

        return False

    def _extract_from_element(self, element: "Tag") -> str:
        parts: List[str] = []

        for child in element.children:
            if isinstance(child, NavigableString):
                text = child.strip()
                if text:
                    parts.append(text)
                continue

            if not hasattr(child, 'name'):
                continue

            tag_name = child.name.lower()

            if tag_name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
                level = int(tag_name[1])
                heading_text = child.get_text(strip=True)
                if heading_text:
                    prefix = "#" * level
                    parts.append(f"\n{prefix} {heading_text}\n")

            elif tag_name == "p":
                text = child.get_text(strip=True)
                if text:
                    parts.append(text)
                parts.append("")

            elif tag_name == "li":
                text = child.get_text(strip=True)
                if text:
                    parts.append(f"- {text}")

            elif tag_name in {"blockquote", "pre", "code"}:
                text = child.get_text()
                if text.strip():
                    parts.append(f"\n{text.strip()}\n")

            elif tag_name in {"br"}:
                parts.append("")

            elif tag_name in {"hr"}:
                parts.append("\n---\n")

            elif tag_name in {"table"}:
                table_text = self._extract_table(child)
                if table_text.strip():
                    parts.append(f"\n{table_text.strip()}\n")

            else:
                nested = self._extract_from_element(child)
                if nested.strip():
                    parts.append(nested)

            if tag_name in SEMANTIC_TAGS and tag_name not in {"li", "br"}:
                parts.append("")

        result = "\n".join(p for p in parts if p is not None)

        if self.clean_whitespace:
            result = re.sub(r'\n{3,}', '\n\n', result)

        return result

    def _extract_table(self, table: "Tag") -> str:
        rows: List[str] = []
        for row in table.find_all("tr"):
            cells = []
            for cell in row.find_all(["td", "th"]):
                cells.append(cell.get_text(strip=True))
            if cells:
                rows.append(" | ".join(cells))
        return "\n".join(rows)
