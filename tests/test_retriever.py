"""retriever 模块测试。"""

import pytest
from unittest.mock import MagicMock

from modules.retriever.base import BaseRetriever
from modules.retriever.vector_retriever import VectorRetriever
from modules.retriever.hybrid_retriever import HybridRetriever
from modules.retriever.factory import RetrieverFactory


class TestBaseRetriever:
    """测试 BaseRetriever 抽象基类。"""

    def test_base_retriever_is_abstract(self):
        """验证 BaseRetriever 无法直接实例化。"""
        with pytest.raises(TypeError):
            BaseRetriever()


class TestVectorRetriever:
    """测试 VectorRetriever。"""

    def test_vector_retriever_search(self, mock_chroma_collection, mock_embedding_client):
        """Mock ChromaDB，验证向量检索返回正确的文档列表格式。"""
        retriever = VectorRetriever(
            collection=mock_chroma_collection,
            embedding_client=mock_embedding_client,
        )
        results = retriever.search("测试查询")
        assert isinstance(results, list)
        assert len(results) == 2
        assert results[0] == "这是测试文档一的内容。"
        assert results[1] == "这是测试文档二的内容。"
        mock_embedding_client.embed_query.assert_called_once_with("测试查询")
        mock_chroma_collection.query.assert_called_once()

    def test_vector_retriever_search_empty(self, mock_chroma_collection_empty, mock_embedding_client):
        """验证无结果时返回空列表。"""
        retriever = VectorRetriever(
            collection=mock_chroma_collection_empty,
            embedding_client=mock_embedding_client,
        )
        results = retriever.search("无结果的查询")
        assert results == []

    def test_vector_retriever_search_custom_top_k(self, mock_chroma_collection, mock_embedding_client):
        """验证自定义 top_k 参数生效。"""
        retriever = VectorRetriever(
            collection=mock_chroma_collection,
            embedding_client=mock_embedding_client,
        )
        retriever.search("测试查询", top_k=10)
        call_args = mock_chroma_collection.query.call_args
        assert call_args[1]["n_results"] == 10


class TestHybridRetriever:
    """测试 HybridRetriever。"""

    def _make_mock_bm25(self, doc_ids, scores=None):
        """创建一个 mock BM25 实例。"""
        bm25 = MagicMock()
        if scores is None:
            scores = [1.5, 0.5]
        bm25.get_scores.return_value = scores
        return bm25

    def test_hybrid_retriever_search(self, mock_chroma_collection, mock_embedding_client):
        """Mock ChromaDB + BM25，测试混合检索融合结果。"""
        tokenized_corpus = [["关键词", "文档"], ["测试", "内容"]]
        doc_ids = ["doc_1", "doc_2"]
        bm25 = self._make_mock_bm25(doc_ids)

        retriever = HybridRetriever(
            collection=mock_chroma_collection,
            embedding_client=mock_embedding_client,
            tokenized_corpus=tokenized_corpus,
            bm25=bm25,
            doc_ids=doc_ids,
            rrf_k=60,
        )
        results = retriever.search("测试查询")
        assert isinstance(results, list)
        assert len(results) <= 5
        for doc in results:
            assert isinstance(doc, str)

    def test_hybrid_retriever_search_empty(self, mock_chroma_collection_empty, mock_embedding_client):
        """验证无文档时返回空列表。"""
        retriever = HybridRetriever(
            collection=mock_chroma_collection_empty,
            embedding_client=mock_embedding_client,
            tokenized_corpus=[],
            bm25=None,
            doc_ids=[],
        )
        results = retriever.search("无结果的查询")
        assert results == []

    def test_hybrid_retriever_bm25_none(self, mock_chroma_collection, mock_embedding_client):
        """验证 BM25 为 None 时仅使用向量检索。"""
        retriever = HybridRetriever(
            collection=mock_chroma_collection,
            embedding_client=mock_embedding_client,
            tokenized_corpus=[],
            bm25=None,
            doc_ids=[],
        )
        results = retriever.search("测试查询")
        assert isinstance(results, list)
        assert len(results) > 0


class TestRetrieverFactory:
    """测试 RetrieverFactory。"""

    def test_factory_creates_vector(self, mock_chroma_collection, mock_embedding_client):
        """验证工厂创建 VectorRetriever 实例。"""
        retriever = RetrieverFactory.create(
            "vector",
            collection=mock_chroma_collection,
            embedding_client=mock_embedding_client,
        )
        assert isinstance(retriever, VectorRetriever)

    def test_factory_creates_hybrid(self, mock_chroma_collection, mock_embedding_client):
        """验证工厂创建 HybridRetriever 实例。"""
        retriever = RetrieverFactory.create(
            "hybrid",
            collection=mock_chroma_collection,
            embedding_client=mock_embedding_client,
            tokenized_corpus=[],
            bm25=None,
            doc_ids=[],
        )
        assert isinstance(retriever, HybridRetriever)

    def test_factory_unknown_type(self):
        """验证传入未知类型时抛出 ValueError。"""
        with pytest.raises(ValueError, match="未知的检索器类型"):
            RetrieverFactory.create("unknown_type")
