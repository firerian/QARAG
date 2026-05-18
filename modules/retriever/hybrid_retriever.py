from typing import List, Optional

import chromadb
import jieba
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from modules.retriever.base import BaseRetriever
from modules.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器：融合向量检索与 BM25 关键词检索，通过 RRF 排序 + 重排序。"""

    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_client: Embeddings,
        tokenized_corpus: List[List[str]],
        bm25: Optional[BM25Okapi],
        doc_ids: List[str],
        rrf_k: Optional[int] = None,
        bm25_k: Optional[int] = None,
        vector_k: Optional[int] = None,
    ) -> None:
        """
        Args:
            collection: ChromaDB 集合实例。
            embedding_client: LangChain embedding 客户端。
            tokenized_corpus: 已分词的语料库，用于 BM25 索引。
            bm25: BM25 索引实例，可为 None（空库时）。
            doc_ids: 文档 ID 列表，与 tokenized_corpus 一一对应。
            rrf_k: 向后兼容参数，若传入则同时用于 bm25_k 和 vector_k。
            bm25_k: BM25 检索的 RRF k 参数，默认 30。
            vector_k: 向量检索的 RRF k 参数，默认 60。
        """
        self._collection = collection
        self._embedding_client = embedding_client
        self._tokenized_corpus = tokenized_corpus
        self._bm25 = bm25
        self._doc_ids = doc_ids
        if rrf_k is not None:
            self._bm25_k = rrf_k
            self._vector_k = rrf_k
        else:
            self._bm25_k = bm25_k if bm25_k is not None else 30
            self._vector_k = vector_k if vector_k is not None else 60

    def _get_metadata(self, doc_id: str) -> dict:
        """获取指定文档的元数据。"""
        try:
            data = self._collection.get(ids=[doc_id], include=["metadatas"])
            if data["metadatas"] and data["metadatas"][0]:
                return data["metadatas"][0]
        except Exception:
            pass
        return {}

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """混合检索：向量检索 + BM25 关键词检索 + RRF 融合 + 结构化增强。

        Args:
            query: 用户查询文本。
            top_k: 返回的文档数量。

        Returns:
            按最终分数排序的文档文本列表。
        """
        candidate_count = max(20, top_k * 4)
        vector_ids = self._vector_search(query, candidate_count)
        bm25_ids = self._bm25_search(query, candidate_count)

        if not vector_ids and not bm25_ids:
            logger.warning("混合检索：向量与BM25均未找到匹配文档")
            return []

        vector_rank = {doc_id: rank for rank, doc_id in enumerate(vector_ids)}
        bm25_rank = {doc_id: rank for rank, doc_id in enumerate(bm25_ids)}

        all_ids = list(set(vector_ids) | set(bm25_ids))
        fused_docs: List[tuple] = []
        for doc_id in all_ids:
            vector_score = 1.0 / (self._vector_k + vector_rank.get(doc_id, candidate_count) + 1)
            bm25_score = 1.0 / (self._bm25_k + bm25_rank.get(doc_id, candidate_count) + 1)
            
            # 结构化增强：识别并提升流程步骤类文档的优先级
            metadata = self._get_metadata(doc_id)
            section_title = metadata.get("section_title", "")
            score_bonus = 0.0
            if "step" in section_title.lower():
                score_bonus = 0.02  # 给予步骤节点微小加分，使其在 top-5 竞争中胜出

            fused_score = vector_score + bm25_score + score_bonus
            fused_docs.append((doc_id, fused_score))

        fused_docs.sort(key=lambda x: x[1], reverse=True)
        sorted_ids = [doc[0] for doc in fused_docs[:top_k]]
        final_docs = self._resolve_documents(sorted_ids)
        return final_docs

    def _vector_search(self, query: str, n_results: int) -> List[str]:
        """执行向量检索，返回文档 ID 列表。"""
        query_embedding = self._embedding_client.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
        )
        ids = results.get("ids", [])
        if ids and ids[0]:
            return list(ids[0])
        return []

    def _bm25_search(self, query: str, n_results: int) -> List[str]:
        """执行 BM25 关键词检索，返回文档 ID 列表。"""
        if self._bm25 is None:
            return []
        tokenized_query = list(jieba.cut(query))
        bm25_scores = self._bm25.get_scores(tokenized_query)
        id_score_pairs = list(zip(self._doc_ids, bm25_scores))
        id_score_pairs.sort(key=lambda x: x[1], reverse=True)
        return [pair[0] for pair in id_score_pairs[:n_results]]

    def _get_bm25_scores(self, query: str) -> dict:
        """获取查询与所有文档的 BM25 原始分数映射（用于调试）。"""
        if self._bm25 is None:
            return {}
        tokenized_query = list(jieba.cut(query))
        scores = self._bm25.get_scores(tokenized_query)
        return {doc_id: float(score) for doc_id, score in zip(self._doc_ids, scores)}

    def _resolve_documents(self, sorted_ids: List[str]) -> List[str]:
        """根据排序后的文档 ID 从 ChromaDB 中提取文档内容。"""
        if not sorted_ids:
            return []
        all_data = self._collection.get(ids=sorted_ids, include=["documents"])
        id_to_doc = {
            doc_id: doc
            for doc_id, doc in zip(all_data["ids"], all_data["documents"])
        }
        return [id_to_doc[doc_id] for doc_id in sorted_ids if doc_id in id_to_doc]
