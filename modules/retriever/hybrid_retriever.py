from typing import List, Optional

import chromadb
import jieba
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from modules.retriever.base import BaseRetriever
from modules.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器：融合向量检索与 BM25 关键词检索，通过 RRF 排序。"""

    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_client: Embeddings,
        tokenized_corpus: List[List[str]],
        bm25: Optional[BM25Okapi],
        doc_ids: List[str],
        rrf_k: int = 60,
    ) -> None:
        """
        Args:
            collection: ChromaDB 集合实例。
            embedding_client: LangChain embedding 客户端。
            tokenized_corpus: 已分词的语料库，用于 BM25 索引。
            bm25: BM25 索引实例，可为 None（空库时）。
            doc_ids: 文档 ID 列表，与 tokenized_corpus 一一对应。
            rrf_k: RRF 融合算法的超参数 k，默认 60。
        """
        self._collection = collection
        self._embedding_client = embedding_client
        self._tokenized_corpus = tokenized_corpus
        self._bm25 = bm25
        self._doc_ids = doc_ids
        self._rrf_k = rrf_k

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """混合检索：向量检索 + BM25 关键词检索 + RRF 融合排序。

        Args:
            query: 用户查询文本。
            top_k: 返回的文档数量。

        Returns:
            按 RRF 融合分数排序的文档文本列表。
        """
        vector_ids = self._vector_search(query, top_k * 3)
        bm25_ids = self._bm25_search(query, top_k * 2)

        if not vector_ids and not bm25_ids:
            logger.warning("混合检索：向量与BM25均未找到匹配文档")
            return []

        sorted_ids = self._rrf_fusion(vector_ids, bm25_ids)[:top_k]
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

    def _rrf_fusion(self, vector_ids: List[str], bm25_ids: List[str]) -> List[str]:
        """使用 RRF（倒数排名融合）算法融合两个检索结果。"""
        fused_scores: dict = {}
        k = self._rrf_k

        for rank, doc_id in enumerate(vector_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        for rank, doc_id in enumerate(bm25_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1.0 / (k + rank + 1)

        return sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

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
