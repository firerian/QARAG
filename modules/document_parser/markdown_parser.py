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


class MarkdownDocumentParser(BaseDocumentParser):
    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
    ):
        super().__init__(chunking_config)
        self.supported_extensions = [".md", ".markdown", ".mdown", ".mkd"]
        self.chunking_strategy = create_chunking_strategy(
            "markdown", self.chunking_config
        )

    def parse(self, file_path: str) -> ProcessResult:
        if not os.path.exists(file_path):
            return ProcessResult(
                success=False,
                errors=[f"File not found: {file_path}"],
            )

        try:
            text = self.extract_text(file_path)
        except Exception as e:
            logger.error("Failed to extract text from Markdown %s: %s", file_path, e)
            return ProcessResult(
                success=False,
                errors=[f"Markdown extraction failed: {str(e)}"],
            )

        if not text or not text.strip():
            return ProcessResult(
                success=False,
                errors=["No content found in Markdown file"],
            )

        base_metadata = {
            "file_name": os.path.basename(file_path),
            "file_path": os.path.abspath(file_path),
            "file_type": "markdown",
        }

        all_chunks = self.chunking_strategy.chunk_text(
            text, metadata=base_metadata
        )

        if not all_chunks:
            return ProcessResult(
                success=False,
                errors=["No chunks generated from Markdown content"],
            )

        return ProcessResult(
            chunks=all_chunks,
            metadata={
                "file_name": os.path.basename(file_path),
                "file_type": "markdown",
                "total_chunks": len(all_chunks),
            },
        )

    def extract_text(self, file_path: str) -> str:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()

        return self._normalize_markdown(content)

    def _normalize_markdown(self, content: str) -> str:
        content = re.sub(r'<!--.*?-->', '', content, flags=re.DOTALL)

        code_blocks, content = self._extract_code_blocks(content)

        content = self._clean_inline_formatting(content)

        content = re.sub(r'^---[\s\S]*?---\s*', '', content, count=1)

        content = self._merge_heading_lines(content)

        content = re.sub(r'!\[.*?\]\(.*?\)', '', content)

        content = self._normalize_links(content)

        content = self._normalize_lists(content)

        content = self._normalize_tables(content)

        content = self._normalize_blockquotes(content)

        content = self._normalize_horizontal_rules(content)

        content = self._restore_code_blocks(content, code_blocks)

        lines = content.splitlines()
        cleaned_lines: List[str] = []
        prev_empty = False
        for line in lines:
            stripped = line.rstrip()
            is_empty = not stripped
            if is_empty:
                if not prev_empty:
                    cleaned_lines.append("")
                prev_empty = True
            else:
                cleaned_lines.append(stripped)
                prev_empty = False

        return "\n".join(cleaned_lines).strip()

    def _merge_heading_lines(self, content: str) -> str:
        result = re.sub(r'\n(#{1,6}\s+[^\n]+)', r'\n\n\1\n', content)
        return result

    def _clean_inline_formatting(self, content: str) -> str:
        content = re.sub(r'~~(.*?)~~', r'\1', content)
        content = re.sub(r'\*\*([^*]+)\*\*', r'\1', content)
        content = re.sub(r'__([^_]+)__', r'\1', content)
        content = re.sub(r'(?<!\*)\*([^*]+)\*(?!\*)', r'\1', content)
        content = re.sub(r'(?<!_)_([^_]+)_(?!_)', r'\1', content)
        content = re.sub(r'`([^`]+)`', r'\1', content)
        return content

    def _normalize_links(self, content: str) -> str:
        def replace_link(match):
            text = match.group(1)
            url = match.group(2)
            if not text:
                return f"<{url}>"
            return text

        content = re.sub(r'\[([^\]]*)\]\(([^)]+)\)', replace_link, content)
        content = re.sub(r'\[([^\]]+)\]\[[^\]]*\]', r'\1', content)
        return content

    def _normalize_lists(self, content: str) -> str:
        lines = content.splitlines()
        result: List[str] = []
        prev_was_list = False
        for line in lines:
            is_unordered = bool(re.match(r'^(\s*)[-*+]\s+', line))
            is_ordered = bool(re.match(r'^(\s*)\d+[.)]\s+', line))
            if (is_unordered or is_ordered) and not prev_was_list:
                result.append("")
            result.append(line)
            prev_was_list = is_unordered or is_ordered
        if prev_was_list:
            result.append("")
        return "\n".join(result)

    def _extract_code_blocks(self, content: str) -> tuple:
        blocks: List[str] = []

        def _save_block(m):
            idx = len(blocks)
            blocks.append(m.group(0).strip())
            return f"\x01\x02\x03{idx}\x03\x02\x01"

        content = re.sub(r'```[\s\S]*?```', _save_block, content)
        return blocks, content

    def _restore_code_blocks(self, content: str, blocks: List[str]) -> str:
        for idx, block in enumerate(blocks):
            content = content.replace(f"\x01\x02\x03{idx}\x03\x02\x01", f"\n{block}\n")
        return content

    def _normalize_tables(self, content: str) -> str:
        lines = content.splitlines()
        result: List[str] = []
        in_table = False
        for line in lines:
            is_table_sep = bool(re.match(r'^\|?[\s\-:|\s]+\|?$', line) and '---' in line)
            is_table_row = bool(re.match(r'^.*\|.*$', line))

            if is_table_sep and in_table:
                continue

            if is_table_row and (in_table or is_table_sep):
                if not in_table:
                    result.append("")
                in_table = True
                cells = [c.strip() for c in line.strip().strip('|').split('|')]
                result.append(" | ".join(cells))
                continue

            if in_table and not is_table_row:
                in_table = False
                result.append("")

            result.append(line)

        return "\n".join(result)

    def _normalize_blockquotes(self, content: str) -> str:
        lines = content.splitlines()
        result: List[str] = []
        prev_was_quote = False
        for line in lines:
            if re.match(r'^>\s*', line):
                cleaned = re.sub(r'^>\s?', '', line)
                if not prev_was_quote:
                    result.append("")
                result.append(cleaned)
                prev_was_quote = True
            else:
                if prev_was_quote:
                    result.append("")
                result.append(line)
                prev_was_quote = False
        return "\n".join(result)

    def _normalize_horizontal_rules(self, content: str) -> str:
        content = re.sub(r'^[-*_]{3,}\s*$', '\n---\n', content, flags=re.MULTILINE)
        return content
