import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Chunk:
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    chunk_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class ChunkingConfig:
    chunk_size: int = 500
    chunk_overlap: int = 50
    preserve_sections: bool = True
    overlap_mode: str = "word"

    def __post_init__(self) -> None:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if self.chunk_overlap < 0:
            raise ValueError("chunk_overlap must be non-negative")
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        if self.overlap_mode not in ("word", "char"):
            raise ValueError("overlap_mode must be 'word' or 'char'")


@dataclass
class ProcessResult:
    chunks: List[Chunk] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    success: bool = True

    @property
    def total_chunks(self) -> int:
        return len(self.chunks)


@dataclass
class ProcessingReport:
    total_files: int = 0
    successful: int = 0
    failed: int = 0
    total_chunks: int = 0
    errors: Dict[str, str] = field(default_factory=dict)
    processor_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    duration_seconds: float = 0.0

    @property
    def success_rate(self) -> float:
        if self.total_files == 0:
            return 1.0
        return self.successful / self.total_files

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_files": self.total_files,
            "successful": self.successful,
            "failed": self.failed,
            "total_chunks": self.total_chunks,
            "success_rate": f"{self.success_rate:.1%}",
            "errors": self.errors,
            "processor_stats": self.processor_stats,
            "duration_seconds": f"{self.duration_seconds:.2f}",
        }


class BaseDocumentParser(ABC):
    def __init__(self, chunking_config: Optional[ChunkingConfig] = None):
        self.chunking_config = chunking_config or ChunkingConfig()
        self.supported_extensions: List[str] = []

    @abstractmethod
    def parse(self, file_path: str) -> ProcessResult:
        """Parse a document file into chunks.

        Args:
            file_path: Path to the document file.

        Returns:
            ProcessResult containing extracted chunks and any errors.
        """
        pass

    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        """Extract raw text from a document without chunking.

        Args:
            file_path: Path to the document file.

        Returns:
            Extracted plain text content.
        """
        pass
