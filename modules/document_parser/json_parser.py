import json
import os
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


class JSONDocumentParser(BaseDocumentParser):
    """JSON/JSONL 格式的问答对文档解析器。

    支持两种 JSON 格式：
    1. JSONL 格式：每行一个 JSON 对象，包含 instruction/output 字段
    2. JSON 数组格式：包含多个问答对对象的数组

    每个问答对会被转换为一个独立的 Chunk，保留问题和答案的完整语义。
    """
    
    def __init__(
        self,
        chunking_config: Optional[ChunkingConfig] = None,
    ):
        super().__init__(chunking_config)
        self.supported_extensions = [".json", ".jsonl"]
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
            qa_pairs = self._load_json(file_path)
        except Exception as e:
            logger.error("Failed to parse JSON %s: %s", file_path, e)
            return ProcessResult(
                success=False,
                errors=[f"JSON parsing failed: {str(e)}"],
            )

        if not qa_pairs:
            return ProcessResult(
                success=False,
                errors=["No QA pairs found in JSON file"],
            )

        base_metadata = {
            "file_name": os.path.basename(file_path),
            "file_path": os.path.abspath(file_path),
            "file_type": "json",
        }

        chunks = self._convert_qa_to_chunks(qa_pairs, base_metadata)

        if not chunks:
            return ProcessResult(
                success=False,
                errors=["No chunks generated from JSON content"],
            )

        return ProcessResult(
            chunks=chunks,
            metadata={
                "file_name": os.path.basename(file_path),
                "file_type": "json",
                "total_chunks": len(chunks),
                "qa_pairs_count": len(qa_pairs),
            },
        )

    def extract_text(self, file_path: str) -> str:
        qa_pairs = self._load_json(file_path)
        texts = []
        for pair in qa_pairs:
            instruction = pair.get("instruction", "")
            output = pair.get("output", "")
            texts.append(f"问题：{instruction}\n答案：{output}")
        return "\n\n".join(texts)

    def _load_json(self, file_path: str) -> List[dict]:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read().strip()

        if not content:
            return []

        first_char = content[0]

        if first_char == "[":
            return self._parse_json_array(content)
        else:
            return self._parse_jsonl(content)

    def _parse_json_array(self, content: str) -> List[dict]:
        data = json.loads(content)
        if not isinstance(data, list):
            raise ValueError("JSON array format expected a list")
        return [item for item in data if isinstance(item, dict)]

    def _parse_jsonl(self, content: str) -> List[dict]:
        qa_pairs = []
        for line_num, line in enumerate(content.splitlines(), 1):
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if isinstance(data, dict):
                    qa_pairs.append(data)
            except json.JSONDecodeError as e:
                logger.warning("Line %d JSON parsing failed, skipped: %s", line_num, e)
        return qa_pairs

    def _convert_qa_to_chunks(
        self, qa_pairs: List[dict], base_metadata: dict
    ) -> List[Chunk]:
        chunks = []
        for idx, pair in enumerate(qa_pairs):
            instruction = pair.get("instruction", "")
            input_text = pair.get("input", "")
            output = pair.get("output", "")

            if not instruction and not output:
                continue

            if input_text:
                content = f"问题：{instruction}\n输入：{input_text}\n答案：{output}"
            else:
                content = f"问题：{instruction}\n答案：{output}"

            chunk_metadata = {
                **base_metadata,
                "qa_index": idx,
                "section_title": instruction[:50] if instruction else "",
                "heading_chain": [instruction[:50] if instruction else ""],
                "section_level": 1,
            }

            chunks.append(Chunk(content=content, metadata=chunk_metadata))

        return chunks
