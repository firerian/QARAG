import argparse
import json
import sys
import os
from typing import Any, List, Optional

from modules import DataProcessor, MyVectorDBConnector, get_llm_answer, embeddingllm
from modules.config import get_config
from modules.logger import get_logger
from modules.document_parser.base import Chunk, ChunkingConfig
from modules.document_parser.batch_processor import BatchProcessor

PARSER_REGISTRY: dict = {}
try:
    from modules.document_parser.pdf_parser import PDFDocumentParser
    PARSER_REGISTRY.update({".pdf": PDFDocumentParser})
except ImportError:
    pass
try:
    from modules.document_parser.html_parser import HTMLDocumentParser
    PARSER_REGISTRY.update({".html": HTMLDocumentParser, ".htm": HTMLDocumentParser})
except ImportError:
    pass
from modules.document_parser.markdown_parser import MarkdownDocumentParser
PARSER_REGISTRY.update({".md": MarkdownDocumentParser, ".markdown": MarkdownDocumentParser,
                        ".mdown": MarkdownDocumentParser, ".mkd": MarkdownDocumentParser})
from modules.document_parser.json_parser import JSONDocumentParser
PARSER_REGISTRY.update({".json": JSONDocumentParser, ".jsonl": JSONDocumentParser})


def ingest_qa(vector_db: Any, file_path: str) -> None:
    logger = get_logger("main")
    logger.info("开始加载问答对数据: %s", file_path)
    try:
        config = get_config()
        processor = DataProcessor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        instructions, outputs = processor.load_qa_json(file_path)
        vector_db.add_documents(outputs, instructions)
        logger.info("问答对数据加载完成，共 %d 条", len(instructions))
    except Exception as e:
        logger.error("加载问答对数据失败: %s", e)
        raise


def ingest_text(vector_db: Any, file_path: str) -> None:
    logger = get_logger("main")
    logger.info("开始加载并切分文本: %s", file_path)
    try:
        config = get_config()
        processor = DataProcessor(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        chunks = processor.load_and_split_text(file_path)
        vector_db.add_documents(chunks)
        logger.info("文本加载并切分完成，共 %d 个片段", len(chunks))
    except Exception as e:
        logger.error("加载文本失败: %s", e)
        raise


def parse_single_document(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Chunk]:
    logger = get_logger("main")
    logger.info("开始解析文档: %s", file_path)

    ext = os.path.splitext(file_path)[1].lower()
    parser_class = PARSER_REGISTRY.get(ext)
    if parser_class is None:
        logger.error("不支持的文件类型: %s (扩展名: %s)", file_path, ext)
        sys.exit(1)

    config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    parser = parser_class(chunking_config=config)
    result = parser.parse(file_path)

    if not result.success:
        for err in result.errors:
            logger.error("解析失败: %s", err)
        sys.exit(1)

    logger.info("解析完成: %d 个片段", len(result.chunks))
    return result.chunks


def parse_directory(
    folder_path: str,
    chunk_size: int,
    chunk_overlap: int,
    max_workers: int,
    timeout: float,
    max_size_mb: float,
    vector_db: Optional[Any] = None,
) -> None:
    logger = get_logger("main")
    logger.info("开始批量解析目录: %s (workers=%d)", folder_path, max_workers)

    chunking_config = ChunkingConfig(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    full_registry = {}
    for ext, parser_cls in PARSER_REGISTRY.items():
        class _ConfiguredParser(parser_cls):
            def __init__(self):
                super().__init__(chunking_config=chunking_config)
        full_registry[ext] = _ConfiguredParser

    processor = BatchProcessor(
        max_workers=max_workers,
        timeout_per_file=timeout,
        max_file_size_mb=max_size_mb,
        parser_registry=full_registry,
    )

    report = processor.process_directory(folder_path)

    if vector_db and report.successful > 0:
        logger.info("正在将 %d 个文件的片段索引到向量库...", report.successful)
        all_files = processor.scan_directory(folder_path)
        for file_path in all_files:
            result = None
            parser_class = PARSER_REGISTRY.get(os.path.splitext(file_path)[1].lower())
            if parser_class:
                parser = parser_class(chunking_config=chunking_config)
                result = parser.parse(file_path)
            if result and result.success and result.chunks:
                contents = [chunk.content for chunk in result.chunks]
                metadatas = [chunk.metadata for chunk in result.chunks]
                vector_db.add_documents(contents, custom_metadatas=metadatas)
        logger.info("索引完成")

    print(processor.generate_report(report))


def interactive_mode(vector_db: Any, retriever_type: str, prompt_strategy: str, top_k: int) -> None:
    logger = get_logger("main")
    logger.info("进入交互模式 (检索器: %s, Prompt策略: %s, top_k: %d)", retriever_type, prompt_strategy, top_k)
    logger.info("输入问题开始对话，输入 'quit' 或 'exit' 退出")

    while True:
        try:
            user_input = input("\n请输入您的问题: ").strip()
        except (EOFError, KeyboardInterrupt):
            logger.info("交互模式已退出")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            logger.info("交互模式已退出")
            break

        logger.info("用户问题: %s", user_input)
        get_llm_answer(vector_db, user_input, retriever_type=retriever_type, prompt_strategy=prompt_strategy, top_k=top_k)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="RAG 问答系统")
    parser.add_argument("--retriever", "-r", choices=["default", "hybrid"], default="hybrid",
                        help="检索器类型 (default: 普通检索, hybrid: 混合检索)")
    parser.add_argument("--query", "-q", type=str, default=None,
                        help="要提问的问题")
    parser.add_argument("--interactive", "-i", action="store_true", default=False,
                        help="进入交互问答模式")
    parser.add_argument("--ingest-qa", type=str, default=None,
                        help="加载并索引 JSONL 格式的问答对文件")
    parser.add_argument("--ingest-text", type=str, default=None,
                        help="加载、切分并索引文本文件")
    parser.add_argument("--rebuild", action="store_true", default=False,
                        help="重建 BM25 索引")
    parser.add_argument("--log-level", type=str, choices=["DEBUG", "INFO", "WARNING", "ERROR"], default=None,
                        help="覆盖日志级别")
    parser.add_argument("--collection", "-c", type=str, default="demo",
                        help="向量库集合名称，默认 demo")
    parser.add_argument("--prompt-strategy", type=str, choices=["strict", "balanced", "creative"], default="strict",
                        help="Prompt 策略，默认 strict")
    parser.add_argument("--top-k", type=int, default=5,
                        help="检索返回的片段数量，默认 5")

    doc_group = parser.add_argument_group("文档解析")
    doc_group.add_argument("--parse-doc", type=str, default=None, metavar="FILE",
                           help="解析单个文档 (PDF/HTML/Markdown) 并输出片段")
    doc_group.add_argument("--parse-doc-index", type=str, default=None, metavar="FILE",
                           help="解析单个文档并同时索引到向量库")
    doc_group.add_argument("--parse-dir", type=str, default=None, metavar="DIR",
                           help="批量解析指定目录下的所有文档并生成报告")
    doc_group.add_argument("--parse-dir-index", type=str, default=None, metavar="DIR",
                           help="批量解析目录并同时索引到向量库")
    doc_group.add_argument("--chunk-size", type=int, default=None,
                           help="文档解析的片段大小（覆盖环境变量 CHUNK_SIZE）")
    doc_group.add_argument("--chunk-overlap", type=int, default=None,
                           help="文档解析的片段重叠量（覆盖环境变量 CHUNK_OVERLAP）")
    doc_group.add_argument("--parse-workers", type=int, default=4,
                           help="批量解析时的并行线程数 (默认: 4)")
    doc_group.add_argument("--parse-timeout", type=float, default=120.0,
                           help="单文件解析超时秒数 (默认: 120)")
    doc_group.add_argument("--parse-max-size", type=float, default=100.0,
                           help="单文件最大大小 (MB) (默认: 100)")
    doc_group.add_argument("--parse-output", type=str, default=None, metavar="FILE",
                           help="将解析结果输出为 JSON 文件")

    return parser


def _output_chunks(chunks: List[Chunk], output_file: Optional[str]) -> None:
    data = [
        {
            "chunk_id": chunk.chunk_id,
            "content": chunk.content,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]
    json_str = json.dumps(data, ensure_ascii=False, indent=2)
    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(json_str)
        print(f"已输出到: {output_file}")
    else:
        print(json_str)


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level

    logger = get_logger("main")
    config = get_config()

    chunk_size = args.chunk_size if args.chunk_size is not None else config.chunk_size
    chunk_overlap = args.chunk_overlap if args.chunk_overlap is not None else config.chunk_overlap

    parse_action = args.parse_doc or args.parse_doc_index or args.parse_dir or args.parse_dir_index
    if parse_action:
        if args.parse_doc:
            chunks = parse_single_document(args.parse_doc, chunk_size, chunk_overlap)
            _output_chunks(chunks, args.parse_output)
        elif args.parse_doc_index:
            if embeddingllm is None:
                logger.error("embeddingllm 未成功加载，请检查依赖")
                return
            chunks = parse_single_document(args.parse_doc_index, chunk_size, chunk_overlap)
            vector_db = MyVectorDBConnector(collection_name=args.collection, client=embeddingllm, config=config)
            contents = [chunk.content for chunk in chunks]
            metadatas = [chunk.metadata for chunk in chunks]
            vector_db.add_documents(contents, custom_metadatas=metadatas)
            logger.info("已解析并索引 %d 个片段到向量库 (collection: %s)", len(chunks), args.collection)
            if args.parse_output:
                _output_chunks(chunks, args.parse_output)
        elif args.parse_dir:
            parse_directory(
                folder_path=args.parse_dir,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_workers=args.parse_workers,
                timeout=args.parse_timeout,
                max_size_mb=args.parse_max_size,
            )
        elif args.parse_dir_index:
            vector_db = MyVectorDBConnector(collection_name=args.collection, client=embeddingllm, config=config)
            parse_directory(
                folder_path=args.parse_dir_index,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                max_workers=args.parse_workers,
                timeout=args.parse_timeout,
                max_size_mb=args.parse_max_size,
                vector_db=vector_db,
            )
        return

    vector_db = MyVectorDBConnector(collection_name=args.collection, client=embeddingllm, config=config)

    has_ingested = False
    if args.ingest_qa:
        ingest_qa(vector_db, args.ingest_qa)
        has_ingested = True
    if args.ingest_text:
        ingest_text(vector_db, args.ingest_text)
        has_ingested = True
    if args.rebuild:
        if not has_ingested:
            logger.info("未摄入新数据，仅重建已有索引")
        vector_db.rebuild_bm25_index()
        logger.info("BM25 索引重建完成")

    if args.interactive:
        interactive_mode(vector_db, args.retriever, args.prompt_strategy, args.top_k)
    elif args.query:
        get_llm_answer(vector_db, args.query, retriever_type=args.retriever, prompt_strategy=args.prompt_strategy, top_k=args.top_k)
    elif has_ingested or args.rebuild:
        logger.info("数据已就绪。使用 -q 提问或 -i 进入交互模式。")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
