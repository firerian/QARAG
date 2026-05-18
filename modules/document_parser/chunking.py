import re
from typing import Dict, List, Optional

try:
    import jieba
    HAS_JIEBA = True
except ImportError:
    jieba = None  # type: ignore
    HAS_JIEBA = False

from modules.document_parser.base import Chunk, ChunkingConfig


class ChunkingStrategy:
    def __init__(self, config: Optional[ChunkingConfig] = None):
        self.config = config or ChunkingConfig()

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        base_metadata = dict(metadata) if metadata else {}

        if self.config.preserve_sections:
            sections = self._split_by_sections(text)
        else:
            sections = [text]

        chunks: List[Chunk] = []
        for section_text in sections:
            section_chunks = self._chunk_section(section_text, base_metadata)
            chunks.extend(section_chunks)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _split_by_sections(self, text: str) -> List[str]:
        section_pattern = re.compile(
            r'(?:(?:^|\n)(?:#{1,6}\s+[^\n]+|\*\*[^*]+\*\*|__[^_]+__)(?:\n|$))',
            re.MULTILINE,
        )

        matches = list(section_pattern.finditer(text))
        if not matches:
            paragraphs = self._split_by_paragraphs(text)
            return [p for p in paragraphs if p.strip()]

        sections: List[str] = []
        start = 0
        for i, match in enumerate(matches):
            section_start = match.start()
            if section_start > start:
                sections.append(text[start:section_start])
            start = match.start()

        if start < len(text):
            sections.append(text[start:])

        result: List[str] = []
        for section in sections:
            if not section.strip():
                continue
            paragraphs = self._split_by_paragraphs(section)
            result.extend([p for p in paragraphs if p.strip()])

        return result

    def _split_by_paragraphs(self, text: str) -> List[str]:
        return re.split(r'\n\s*\n', text)

    def _chunk_section(
        self,
        text: str,
        base_metadata: Dict,
    ) -> List[Chunk]:
        if self.config.overlap_mode == "char":
            return self._chunk_by_chars(text, base_metadata)
        else:
            return self._chunk_by_words(text, base_metadata)

    def _chunk_by_words(
        self,
        text: str,
        base_metadata: Dict,
    ) -> List[Chunk]:
        if HAS_JIEBA:
            words = [w for w in jieba.lcut(text) if w.strip()]
        else:
            words = text.split()
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if not words:
            return []

        if len(words) <= chunk_size:
            return [Chunk(content=text, metadata=dict(base_metadata))]

        chunks: List[Chunk] = []
        start = 0
        step = max(1, chunk_size - overlap)

        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = " ".join(chunk_words)
            chunk_meta = dict(base_metadata)
            chunk_meta["chunk_start_word"] = start
            chunk_meta["chunk_end_word"] = end
            chunks.append(Chunk(content=chunk_text, metadata=chunk_meta))
            start += step
            if start >= len(words):
                break

        return chunks

    def _chunk_by_chars(
        self,
        text: str,
        base_metadata: Dict,
    ) -> List[Chunk]:
        chunk_size = self.config.chunk_size
        overlap = self.config.chunk_overlap

        if len(text) <= chunk_size:
            return [Chunk(content=text, metadata=dict(base_metadata))]

        chunks: List[Chunk] = []
        start = 0
        step = max(1, chunk_size - overlap)

        while start < len(text):
            end = start + chunk_size
            chunk_text = text[start:end]
            chunk_meta = dict(base_metadata)
            chunk_meta["char_start"] = start
            chunk_meta["char_end"] = min(end, len(text))
            chunks.append(Chunk(content=chunk_text, metadata=chunk_meta))
            start += step
            if start >= len(text):
                break

        return chunks


class MarkdownChunkingStrategy(ChunkingStrategy):
    """Markdown 专用分块策略：基于标题层级构建语义单元。

    每个 chunk 保留其所属的完整标题链作为前缀，确保：
    - 标题与内容不会被拆分到不同 chunk
    - 检索时可通过标题上下文精确匹配
    - metadata 中记录 section_title / heading_chain / section_level
    """

    _HEADING_PATTERN = re.compile(r'^(#{1,6})\s+(.+)$')

    def chunk_text(
        self,
        text: str,
        metadata: Optional[Dict] = None,
    ) -> List[Chunk]:
        if not text or not text.strip():
            return []

        base_metadata = dict(metadata) if metadata else {}
        sections = self._parse_heading_sections(text)

        chunks: List[Chunk] = []
        for section in sections:
            section_chunks = self._chunk_section_with_headings(section, base_metadata)
            chunks.extend(section_chunks)

        for idx, chunk in enumerate(chunks):
            chunk.metadata["chunk_index"] = idx
            chunk.metadata["total_chunks"] = len(chunks)

        return chunks

    def _parse_heading_sections(self, text: str) -> List[dict]:
        lines = text.splitlines()
        sections: List[dict] = []
        heading_stack: List[tuple] = []
        content_buf: List[str] = []
        has_started = False
        in_code_block = False
        code_fence_char = None

        def _flush() -> None:
            nonlocal content_buf
            if not has_started:
                return
            content = "\n".join(content_buf).strip()
            if heading_stack:
                chain_titles = [h[1] for h in heading_stack]
                prefix = "\n".join(
                    f"{'#' * h[0]} {h[1]}" for h in heading_stack
                )
                sections.append({
                    "level": heading_stack[-1][0],
                    "title": heading_stack[-1][1],
                    "heading_chain": chain_titles,
                    "heading_prefix": prefix,
                    "content": content,
                })
            elif content:
                sections.append({
                    "level": 0,
                    "title": "",
                    "heading_chain": [],
                    "heading_prefix": "",
                    "content": content,
                })
            content_buf = []

        for line in lines:
            stripped = line.strip()
            if stripped.startswith("```"):
                fence_char = stripped[0]
                if not in_code_block:
                    in_code_block = True
                    code_fence_char = fence_char
                elif stripped.startswith(code_fence_char * 3):
                    in_code_block = False
                    code_fence_char = None
                content_buf.append(line)
                has_started = True
                continue

            if in_code_block:
                content_buf.append(line)
                has_started = True
                continue

            m = self._HEADING_PATTERN.match(line)
            if m:
                _flush()
                level = len(m.group(1))
                title = m.group(2).strip()
                while heading_stack and heading_stack[-1][0] >= level:
                    heading_stack.pop()
                heading_stack.append((level, title))
                has_started = True
            else:
                content_buf.append(line)
                has_started = True

        _flush()
        return sections

    def _chunk_section_with_headings(
        self,
        section: dict,
        base_metadata: Dict,
    ) -> List[Chunk]:
        heading_prefix = section["heading_prefix"]
        content = section["content"]

        if not content.strip():
            return []

        full_text = f"{heading_prefix}\n\n{content}" if heading_prefix else content
        full_words = self._tokenize(full_text)

        chunk_meta_extra = {
            "section_title": section["title"],
            "heading_chain": section["heading_chain"],
            "section_level": section["level"],
        }

        if len(full_words) <= self.config.chunk_size:
            return [Chunk(
                content=full_text,
                metadata={**base_metadata, **chunk_meta_extra},
            )]

        paragraphs = self._split_by_paragraphs(content)
        chunks: List[Chunk] = []

        for para in paragraphs:
            if not para.strip():
                continue
            para_text = f"{heading_prefix}\n\n{para.strip()}" if heading_prefix else para.strip()
            para_words = self._tokenize(para_text)

            if len(para_words) <= self.config.chunk_size:
                chunks.append(Chunk(
                    content=para_text,
                    metadata={**base_metadata, **chunk_meta_extra},
                ))
            else:
                step = max(1, self.config.chunk_size - self.config.chunk_overlap)
                for i in range(0, len(para_words), step):
                    end = min(i + self.config.chunk_size, len(para_words))
                    sub_words = para_words[i:end]
                    sub_text = " ".join(sub_words)
                    if heading_prefix:
                        sub_text = f"{heading_prefix}\n\n{sub_text}"
                    chunks.append(Chunk(
                        content=sub_text,
                        metadata={
                            **base_metadata,
                            **chunk_meta_extra,
                            "chunk_start_word": i,
                            "chunk_end_word": end,
                        },
                    ))

        return chunks

    def _tokenize(self, text: str) -> List[str]:
        if HAS_JIEBA:
            return [w for w in jieba.lcut(text) if w.strip()]
        return text.split()


DEFAULT_CONFIGS: Dict[str, ChunkingConfig] = {
    "pdf": ChunkingConfig(
        chunk_size=400,
        chunk_overlap=50,
        preserve_sections=True,
        overlap_mode="word",
    ),
    "html": ChunkingConfig(
        chunk_size=500,
        chunk_overlap=50,
        preserve_sections=True,
        overlap_mode="word",
    ),
    "markdown": ChunkingConfig(
        chunk_size=800,
        chunk_overlap=80,
        preserve_sections=True,
        overlap_mode="word",
    ),
}


def create_chunking_strategy(
    doc_type: Optional[str] = None,
    config: Optional[ChunkingConfig] = None,
) -> ChunkingStrategy:
    if config is not None:
        if doc_type == "markdown":
            return MarkdownChunkingStrategy(config)
        return ChunkingStrategy(config)
    if doc_type == "markdown":
        return MarkdownChunkingStrategy(DEFAULT_CONFIGS.get("markdown"))
    if doc_type and doc_type in DEFAULT_CONFIGS:
        return ChunkingStrategy(DEFAULT_CONFIGS[doc_type])
    return ChunkingStrategy()
