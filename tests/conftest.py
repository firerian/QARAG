"""共享 fixtures，供所有测试模块使用。

本文件在 pytest 收集测试用例之前被加载，因此在此处 mock
所有运行时环境中未安装的第三方依赖，并设置必要的环境变量，
确保导入路径畅通。
"""

import sys
import os
from unittest.mock import MagicMock, Mock, patch

_MODULE_MOCKS = {
    "langchain_ollama": MagicMock(),
    "langchain_ollama.OllamaEmbeddings": MagicMock(),
    "langchain_openai": MagicMock(),
    "langchain_openai.ChatOpenAI": MagicMock(),
    "dotenv": MagicMock(),
    "tenacity": MagicMock(),
    "tenacity.retry": MagicMock(),
    "chromadb": MagicMock(),
    "chromadb.config": MagicMock(),
    "rank_bm25": MagicMock(),
    "jieba": MagicMock(),
}

for _mod_name, _mod_mock in _MODULE_MOCKS.items():
    if _mod_name not in sys.modules:
        sys.modules[_mod_name] = _mod_mock

os.environ.setdefault("LLM_API_KEY", "test-key-for-collection")
os.environ.setdefault("LLM_MODEL", "test-model")
os.environ.setdefault("LLM_BASE_URL", "https://test.example.com")

import pytest


@pytest.fixture
def mock_config():
    """返回一个所有必填字段都已填充的 Mock Config 对象。"""
    config = MagicMock()
    config.llm_api_key = "test-api-key-12345"
    config.llm_model = "gpt-4-test"
    config.llm_base_url = "https://test-api.example.com/v1"
    config.temperature = 0.0
    config.max_tokens = 2048
    config.vector_db_path = "./test_chroma_data"
    config.embedding_model = "bge-m3:567m"
    config.chunk_size = 500
    config.chunk_overlap = 50
    config.top_k = 5
    config.rrf_k = 60
    config.log_level = "INFO"
    config.dedup_strategy = "skip"
    config.prompt_strategy = "strict"
    return config


@pytest.fixture
def mock_chroma_collection():
    """返回一个 Mock ChromaDB collection，其 query/get 方法返回可预测的结果。"""
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [["doc_1", "doc_2"]],
        "documents": [["这是测试文档一的内容。", "这是测试文档二的内容。"]],
        "metadatas": [[{"source": "test"}, {"source": "test"}]],
        "distances": [[0.12, 0.34]],
    }
    collection.get.return_value = {
        "ids": ["doc_1", "doc_2"],
        "documents": ["这是测试文档一的内容。", "这是测试文档二的内容。"],
    }
    collection.count.return_value = 2
    return collection


@pytest.fixture
def mock_chroma_collection_empty():
    """返回一个返回空结果的 Mock ChromaDB collection。"""
    collection = MagicMock()
    collection.query.return_value = {
        "ids": [[]],
        "documents": [[]],
        "metadatas": [[]],
        "distances": [[]],
    }
    collection.get.return_value = {
        "ids": [],
        "documents": [],
    }
    collection.count.return_value = 0
    return collection


@pytest.fixture
def mock_embedding_client():
    """返回一个 Mock embedding 客户端，embed_query 返回固定维度的 dummy 向量。"""
    client = MagicMock()
    client.embed_query.return_value = [0.1] * 1024
    client.embed_documents.return_value = [[0.1] * 1024, [0.2] * 1024]
    return client


@pytest.fixture
def mock_llm():
    """返回一个 Mock LLM，invoke 返回可预测的响应。"""
    llm = MagicMock()
    response = MagicMock()
    response.content = "这是一个模拟的 AI 回答。"
    response.response_metadata = {
        "token_usage": {
            "prompt_tokens": 150,
            "completion_tokens": 50,
            "total_tokens": 200,
        }
    }
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def clean_env(monkeypatch):
    """清空相关环境变量，确保测试环境干净。"""
    env_vars = [
        "LLM_API_KEY", "LLM_MODEL", "LLM_BASE_URL",
        "TEMPERATURE", "MAX_TOKENS", "VECTOR_DB_PATH",
        "EMBEDDING_MODEL", "CHUNK_SIZE", "CHUNK_OVERLAP",
        "TOP_K", "RRF_K", "LOG_LEVEL", "DEDUP_STRATEGY", "PROMPT_STRATEGY",
    ]
    for var in env_vars:
        monkeypatch.delenv(var, raising=False)
    yield
