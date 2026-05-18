"""集成测试：mock 完整 RAG 流水线。"""

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from modules.data_processor import DataProcessor
from modules.prompts import build_prompt
from modules.retriever.vector_retriever import VectorRetriever
from modules.retriever.hybrid_retriever import HybridRetriever


class TestRAGPipeline:
    """测试完整 RAG 流水线：数据加载 → 向量化 → 检索 → LLM 回答。"""

    def test_rag_pipeline_data_loading(self):
        """验证 DataProcessor 加载并切分数据的流程。"""
        import tempfile
        import os

        content = "这是文档A的内容，包含重要信息。\n\n这是文档B的内容，也包含关键数据。"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            processor = DataProcessor(chunk_size=100, chunk_overlap=20)
            chunks = processor.load_and_split_text(temp_path)
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            combined = "".join(chunks)
            assert "文档A" in combined or "文档B" in combined
        finally:
            os.unlink(temp_path)

    def test_rag_pipeline_vector_search(self, mock_chroma_collection, mock_embedding_client):
        """验证向量检索流程：query → embedding → chromadb → results。"""
        retriever = VectorRetriever(
            collection=mock_chroma_collection,
            embedding_client=mock_embedding_client,
        )
        results = retriever.search("RAG 是什么？")
        assert len(results) >= 1
        mock_embedding_client.embed_query.assert_called_once()
        mock_chroma_collection.query.assert_called_once()

    def test_rag_pipeline_hybrid_search(self, mock_chroma_collection, mock_embedding_client):
        """验证混合检索流程：向量 + BM25 + RRF 融合。"""
        mock_bm25 = MagicMock()
        mock_bm25.get_scores.return_value = [1.0, 0.5]

        retriever = HybridRetriever(
            collection=mock_chroma_collection,
            embedding_client=mock_embedding_client,
            tokenized_corpus=[["doc", "a"], ["doc", "b"]],
            bm25=mock_bm25,
            doc_ids=["doc_1", "doc_2"],
            rrf_k=60,
        )
        results = retriever.search("RAG 混合检索")
        assert isinstance(results, list)
        assert len(results) >= 1

    def test_rag_pipeline_prompt_building(self):
        """验证从检索结果到 Prompt 构建的流程。"""
        retrieved_docs = ["文档A：RAG 是检索增强生成技术。", "文档B：RAG 结合了检索和生成。"]
        contents = "\n".join(retrieved_docs)
        user_query = "什么是 RAG？"

        prompt = build_prompt(strategy="strict", contents=contents, user_query=user_query)
        assert "文档A" in prompt
        assert "文档B" in prompt
        assert user_query in prompt

    def test_rag_pipeline_empty_retrieval(self):
        """验证检索结果为空时 Prompt 包含占位文本。"""
        prompt = build_prompt(strategy="strict", contents="", user_query="什么是 XYZ？")
        assert "无相关已知信息" in prompt
        assert "什么是 XYZ？" in prompt

    def test_rag_pipeline_llm_answer_flow(self):
        """Mock LLM，验证 get_llm_answer 的完整调用流程。"""
        mock_vector_db = MagicMock()
        mock_vector_db.hybrid_search.return_value = [
            "检索到的相关文档一。",
            "检索到的相关文档二。",
        ]

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "根据检索到的文档，答案是模拟的 AI 回复。"
        mock_response.response_metadata = {
            "token_usage": {
                "prompt_tokens": 100,
                "completion_tokens": 30,
                "total_tokens": 130,
            }
        }
        mock_llm.invoke.return_value = mock_response

        with patch("modules.llm_handler.llm", mock_llm):
            with patch("modules.llm_handler.build_prompt") as mock_build:
                mock_build.return_value = "构建好的 Prompt 文本"
                from modules.llm_handler import get_llm_answer

                get_llm_answer(
                    mock_vector_db,
                    "测试问题",
                    llm=mock_llm,
                    retriever_type="hybrid",
                    prompt_strategy="strict",
                )

                mock_vector_db.hybrid_search.assert_called_once_with("测试问题", 5)
                mock_build.assert_called_once()
                assert mock_build.call_args[1]["strategy"] == "strict"
                assert mock_build.call_args[1]["user_query"] == "测试问题"

    def test_rag_pipeline_default_search(self):
        """验证 default 检索类型的调用流程。"""
        mock_vector_db = MagicMock()
        mock_vector_db.search.return_value = {
            "documents": [["文档A", "文档B"]],
            "ids": [["id1", "id2"]],
        }

        mock_llm = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "回答内容"
        mock_response.response_metadata = {}
        mock_llm.invoke.return_value = mock_response

        with patch("modules.llm_handler.llm", mock_llm):
            with patch("modules.llm_handler.build_prompt") as mock_build:
                mock_build.return_value = "Prompt 内容"
                from modules.llm_handler import get_llm_answer

                result = get_llm_answer(
                    mock_vector_db,
                    "测试",
                    llm=mock_llm,
                    retriever_type="default",
                )

                mock_vector_db.search.assert_called_once_with("测试", 5)

    def test_rag_pipeline_invalid_retriever_type(self):
        """验证非法 retriever_type 抛出 ValueError。"""
        mock_vector_db = MagicMock()
        mock_llm = MagicMock()

        with patch("modules.llm_handler.llm", mock_llm):
            from modules.llm_handler import get_llm_answer
            with pytest.raises(ValueError, match="不支持的 retriever_type"):
                get_llm_answer(
                    mock_vector_db,
                    "测试问题",
                    llm=mock_llm,
                    retriever_type="invalid_type",
                )

    def test_rag_pipeline_llm_failure_graceful(self):
        """验证 LLM 调用失败时返回降级消息。"""
        mock_vector_db = MagicMock()
        mock_vector_db.hybrid_search.return_value = ["文档内容"]

        mock_llm = MagicMock()
        mock_llm.invoke.side_effect = Exception("LLM 服务不可用")

        def _real_invoke_llm(_llm, prompt):
            return _llm.invoke(prompt)

        with patch("modules.llm_handler.llm", mock_llm):
            with patch("modules.llm_handler._invoke_llm", side_effect=_real_invoke_llm):
                from modules.llm_handler import get_llm_answer

                result = get_llm_answer(
                    mock_vector_db,
                    "测试问题",
                    llm=mock_llm,
                    retriever_type="hybrid",
                )
                assert "AI 服务暂时不可用" in result
