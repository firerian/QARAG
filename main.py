import argparse
import sys
import os
from typing import Any

from modules import DataProcessor, MyVectorDBConnector, get_llm_answer, embeddingllm
from modules.config import get_config
from modules.logger import get_logger


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


def interactive_mode(vector_db: Any, retriever_type: str, prompt_strategy: str) -> None:
    logger = get_logger("main")
    logger.info("进入交互模式 (检索器: %s, Prompt策略: %s)", retriever_type, prompt_strategy)
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
        get_llm_answer(vector_db, user_input, retriever_type=retriever_type, prompt_strategy=prompt_strategy)


def main() -> None:
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

    args = parser.parse_args()

    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level

    logger = get_logger("main")
    config = get_config()
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
        interactive_mode(vector_db, args.retriever, args.prompt_strategy)
    elif args.query:
        get_llm_answer(vector_db, args.query, retriever_type=args.retriever, prompt_strategy=args.prompt_strategy)
    elif has_ingested or args.rebuild:
        logger.info("数据已就绪。使用 -q 提问或 -i 进入交互模式。")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
