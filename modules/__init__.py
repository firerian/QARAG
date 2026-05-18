try:
    from .llm_handler import llm, embeddingllm, get_llm_answer, get_llm_answer2, check_embedding_health
except ImportError:
    llm = None
    embeddingllm = None
    get_llm_answer = None
    get_llm_answer2 = None
    check_embedding_health = None

try:
    from .vector_db import MyVectorDBConnector
except ImportError:
    MyVectorDBConnector = None

from .data_processor import DataProcessor
from .config import Config, get_config
from .logger import get_logger, setup_logger
from .prompts import PROMPT_TEMPLATES, get_prompt_template, build_prompt

try:
    from .retriever import BaseRetriever, VectorRetriever, HybridRetriever, RetrieverFactory
except ImportError:
    BaseRetriever = None
    VectorRetriever = None
    HybridRetriever = None
    RetrieverFactory = None

from .document_parser import (
    Chunk,
    ChunkingConfig,
    ProcessingReport,
    ProcessResult,
    BaseDocumentParser,
    ChunkingStrategy,
    create_chunking_strategy,
    PDFDocumentParser,
    HTMLDocumentParser,
    MarkdownDocumentParser,
    BatchProcessor,
    BatchProgress,
)
