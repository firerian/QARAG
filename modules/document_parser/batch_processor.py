import os
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as FutureTimeoutError
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Type

from modules.document_parser.base import (
    BaseDocumentParser,
    Chunk,
    ChunkingConfig,
    ProcessResult,
    ProcessingReport,
)
from modules.document_parser.pdf_parser import PDFDocumentParser, PDF_SUPPORT
from modules.document_parser.html_parser import HTMLDocumentParser, HTML_SUPPORT
from modules.document_parser.markdown_parser import MarkdownDocumentParser
from modules.document_parser.json_parser import JSONDocumentParser
from modules.logger import get_logger

logger = get_logger(__name__)

SUPPORTED_EXTENSIONS: Set[str] = {
    ".pdf",
    ".html", ".htm",
    ".md", ".markdown", ".mdown", ".mkd",
    ".json", ".jsonl",
}

DEFAULT_PARSER_REGISTRY: Dict[str, Type[BaseDocumentParser]] = {}


def _build_default_registry() -> Dict[str, Type[BaseDocumentParser]]:
    registry: Dict[str, Type[BaseDocumentParser]] = {}
    if PDF_SUPPORT:
        registry[".pdf"] = PDFDocumentParser
    if HTML_SUPPORT:
        registry[".html"] = HTMLDocumentParser
        registry[".htm"] = HTMLDocumentParser
    registry[".md"] = MarkdownDocumentParser
    registry[".markdown"] = MarkdownDocumentParser
    registry[".mdown"] = MarkdownDocumentParser
    registry[".mkd"] = MarkdownDocumentParser
    registry[".json"] = JSONDocumentParser
    registry[".jsonl"] = JSONDocumentParser
    return registry


DEFAULT_PARSER_REGISTRY = _build_default_registry()


@dataclass
class BatchProgress:
    total_files: int = 0
    processed: int = 0
    successful: int = 0
    failed: int = 0
    total_chunks: int = 0
    start_time: float = 0.0
    current_file: str = ""
    failed_files: Dict[str, str] = field(default_factory=dict)
    processor_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    @property
    def elapsed_seconds(self) -> float:
        if self.start_time == 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def estimated_remaining_seconds(self) -> Optional[float]:
        if self.processed == 0 or self.total_files == 0 or self.start_time == 0:
            return None
        avg_time = self.elapsed_seconds / self.processed
        remaining = self.total_files - self.processed
        return avg_time * remaining

    def get_summary(self) -> str:
        lines = [
            f"Progress: {self.processed}/{self.total_files} files",
            f"  Successful: {self.successful}",
            f"  Failed: {self.failed}",
            f"  Total chunks: {self.total_chunks}",
            f"  Elapsed: {self.elapsed_seconds:.1f}s",
        ]
        eta = self.estimated_remaining_seconds
        if eta is not None:
            lines.append(f"  ETA: {eta:.1f}s")
        if self.current_file:
            lines.append(f"  Current: {self.current_file}")
        return "\n".join(lines)


class BatchProcessor:
    def __init__(
        self,
        max_workers: int = 4,
        timeout_per_file: float = 120.0,
        max_file_size_mb: float = 100.0,
        parser_registry: Optional[Dict[str, Type[BaseDocumentParser]]] = None,
        on_progress: Optional[Callable[[BatchProgress], None]] = None,
    ):
        self.max_workers = max_workers
        self.timeout_per_file = timeout_per_file
        self.max_file_size_bytes = int(max_file_size_mb * 1024 * 1024)
        self.parser_registry = parser_registry or dict(DEFAULT_PARSER_REGISTRY)
        self.on_progress = on_progress

    def scan_directory(self, folder_path: str) -> List[str]:
        files: List[str] = []
        for root, dirs, filenames in os.walk(folder_path):
            for filename in filenames:
                ext = os.path.splitext(filename)[1].lower()
                if ext in self.parser_registry:
                    files.append(os.path.join(root, filename))
        return sorted(files)

    def process_directory(self, folder_path: str) -> ProcessingReport:
        if not os.path.isdir(folder_path):
            raise ValueError(f"Not a directory: {folder_path}")

        files = self.scan_directory(folder_path)

        if not files:
            logger.warning("No supported files found in: %s", folder_path)
            return ProcessingReport(total_files=0)

        progress = BatchProgress(
            total_files=len(files),
            start_time=time.time(),
        )

        logger.info(
            "Starting batch processing: %d files, %d workers",
            len(files), self.max_workers,
        )

        all_chunks: List[Chunk] = []
        report = ProcessingReport(total_files=len(files))

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_map: Dict[Any, str] = {}
            for file_path in files:
                future = executor.submit(self._process_single_file, file_path)
                future_map[future] = file_path

            for future in as_completed(future_map):
                file_path = future_map[future]
                progress.current_file = os.path.basename(file_path)
                progress.processed += 1

                try:
                    result = future.result(timeout=self.timeout_per_file + 10)

                    ext = os.path.splitext(file_path)[1].lower()
                    report.processor_stats.setdefault(ext, {"success": 0, "failed": 0})

                    if result.success:
                        progress.successful += 1
                        progress.total_chunks += result.total_chunks
                        all_chunks.extend(result.chunks)
                        report.processor_stats[ext]["success"] += 1
                        logger.debug("Processed: %s -> %d chunks", file_path, result.total_chunks)
                    else:
                        progress.failed += 1
                        error_msg = "; ".join(result.errors)
                        progress.failed_files[file_path] = error_msg
                        report.errors[file_path] = error_msg
                        report.processor_stats[ext]["failed"] += 1
                        logger.warning("Failed to process %s: %s", file_path, error_msg)

                except FutureTimeoutError:
                    progress.failed += 1
                    msg = f"Timeout after {self.timeout_per_file}s"
                    progress.failed_files[file_path] = msg
                    report.errors[file_path] = msg
                    ext = os.path.splitext(file_path)[1].lower()
                    report.processor_stats.setdefault(ext, {"success": 0, "failed": 0})
                    report.processor_stats[ext]["failed"] += 1
                    logger.error("Timeout processing: %s", file_path)

                except Exception as e:
                    progress.failed += 1
                    msg = f"{type(e).__name__}: {str(e)}"
                    progress.failed_files[file_path] = msg
                    report.errors[file_path] = msg
                    ext = os.path.splitext(file_path)[1].lower()
                    report.processor_stats.setdefault(ext, {"success": 0, "failed": 0})
                    report.processor_stats[ext]["failed"] += 1
                    logger.error("Error processing %s: %s", file_path, e)

                if self.on_progress:
                    try:
                        self.on_progress(progress)
                    except Exception:
                        pass

        report.successful = progress.successful
        report.failed = progress.failed
        report.total_chunks = progress.total_chunks
        report.duration_seconds = time.time() - progress.start_time

        logger.info(
            "Batch processing complete: %d/%d successful, %d chunks, %.1fs",
            report.successful, report.total_files, report.total_chunks, report.duration_seconds,
        )

        return report

    def _process_single_file(self, file_path: str) -> ProcessResult:
        try:
            ext = os.path.splitext(file_path)[1].lower()

            if not os.path.exists(file_path):
                return ProcessResult(
                    success=False,
                    errors=[f"File not found: {file_path}"],
                )

            file_size = os.path.getsize(file_path)
            if file_size > self.max_file_size_bytes:
                return ProcessResult(
                    success=False,
                    errors=[
                        f"File exceeds size limit: {file_size / 1024 / 1024:.1f}MB > "
                        f"{self.max_file_size_bytes / 1024 / 1024:.1f}MB"
                    ],
                )

            if file_size == 0:
                return ProcessResult(
                    success=False,
                    errors=[f"File is empty: {file_path}"],
                )

            parser_class = self.parser_registry.get(ext)
            if parser_class is None:
                return ProcessResult(
                    success=False,
                    errors=[f"No parser available for extension: {ext}"],
                )

            parser = parser_class()
            return parser.parse(file_path)

        except Exception as e:
            return ProcessResult(
                success=False,
                errors=[f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"],
            )

    def generate_report(self, report: ProcessingReport) -> str:
        lines = [
            "=" * 60,
            "        DOCUMENT PROCESSING REPORT",
            "=" * 60,
            "",
            f"Total files detected:   {report.total_files}",
            f"Successfully processed: {report.successful}",
            f"Failed to process:      {report.failed}",
            f"Success rate:           {report.success_rate:.1%}",
            f"Total chunks generated: {report.total_chunks}",
            f"Total duration:         {report.duration_seconds:.2f}s",
            "",
        ]

        if report.processor_stats:
            lines.append("Per-format statistics:")
            lines.append("-" * 40)
            for ext, stats in sorted(report.processor_stats.items()):
                lines.append(
                    f"  {ext}: {stats.get('success', 0)} success, "
                    f"{stats.get('failed', 0)} failed"
                )
            lines.append("")

        if report.errors:
            lines.append(f"Errors encountered ({len(report.errors)} files):")
            lines.append("-" * 40)
            for file_path, error in report.errors.items():
                lines.append(f"  {os.path.basename(file_path)}:")
                for err_line in error.split("\n")[:3]:
                    lines.append(f"    {err_line}")
                lines.append("")
        else:
            lines.append("No errors encountered.")

        lines.append("=" * 60)
        return "\n".join(lines)
