from typing import List, Optional

import chromadb
import jieba
from langchain_core.embeddings import Embeddings
from rank_bm25 import BM25Okapi

from modules.retriever.base import BaseRetriever
from modules.logger import get_logger
from modules.config import Config, get_config

logger = get_logger(__name__)


class HybridRetriever(BaseRetriever):
    """混合检索器：融合向量检索与 BM25 关键词检索，通过 RRF 排序 + 元数据增强重排序。"""

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
        config: Optional[Config] = None,
    ) -> None:
        """
        Args:
            collection: ChromaDB 集合实例。
            embedding_client: LangChain embedding 客户端。
            tokenized_corpus: 已分词的语料库，用于 BM25 索引。
            bm25: BM25 索引实例，可为 None（空库时）。
            doc_ids: 文档 ID 列表，与 tokenized_corpus 一一对应。
            rrf_k: 向后兼容参数，若传入则同时用于 bm25_k 和 vector_k。
            bm25_k: BM25 检索的 RRF k 参数，默认 60。
            vector_k: 向量检索的 RRF k 参数，默认 60。
            config: 应用配置实例，用于读取元数据增强相关参数。
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
            self._bm25_k = bm25_k if bm25_k is not None else 60
            self._vector_k = vector_k if vector_k is not None else 60
        
        self._config = config or get_config()

    def _get_metadata(self, doc_id: str) -> dict:
        """获取指定文档的元数据。"""
        try:
            data = self._collection.get(ids=[doc_id], include=["metadatas"])
            if data["metadatas"] and data["metadatas"][0]:
                return data["metadatas"][0]
        except Exception:
            pass
        return {}

    def _batch_get_metadata(self, doc_ids: List[str]) -> dict:
        """批量获取多个文档的元数据，提升性能。
        
        Args:
            doc_ids: 文档ID列表。
            
        Returns:
            文档ID到元数据的映射字典。
        """
        if not doc_ids:
            return {}
        
        try:
            data = self._collection.get(ids=doc_ids, include=["metadatas"])
            return {
                doc_id: (meta if meta else {})
                for doc_id, meta in zip(data["ids"], data["metadatas"])
            }
        except Exception as e:
            logger.warning("批量获取元数据失败，回退到逐个查询: %s", str(e))
            result = {}
            for doc_id in doc_ids:
                result[doc_id] = self._get_metadata(doc_id)
            return result

    def _calculate_metadata_bonus(self, query: str, metadata: dict) -> float:
        """基于文档元数据计算相关性加分。
        
        通过多维度分析文档的结构特征、标题层级、关键词匹配度等，
        为更相关、更有价值的文档提供更高的加分。
        
        Args:
            query: 用户查询文本。
            metadata: 文档元数据字典。
            
        Returns:
            相关性加分分数（0.0 - metadata_bonus_max范围内）。
        """
        # 检查是否启用元数据增强
        if not self._config.metadata_boost_enabled:
            return 0.0
        
        bonus = 0.0
        
        # 提取元数据字段
        section_title = metadata.get("section_title", "")
        heading_chain = metadata.get("heading_chain", [])
        section_level = metadata.get("section_level", 999)
        source = metadata.get("source", "")
        
        # 1. 流程步骤识别（最高优先级）
        # 匹配各种形式的步骤表述
        step_patterns = ["step", "步骤", "流程", "工作流", "workflow"]
        step_bonus = self._config.metadata_step_bonus
        
        # 检查标题是否包含步骤关键词
        for pattern in step_patterns:
            if pattern in section_title.lower():
                bonus += step_bonus
                break
        
        # 检查标题链是否包含步骤相关父级
        for heading in heading_chain:
            for pattern in step_patterns:
                if pattern in heading.lower():
                    bonus += step_bonus * 0.5
                    break
        
        # 2. 层级结构加分（浅层级通常更重要）
        if section_level == 1:
            bonus += self._config.metadata_level_bonus_1
        elif section_level == 2:
            bonus += self._config.metadata_level_bonus_2
        elif section_level == 3:
            bonus += self._config.metadata_level_bonus_3
        
        # 3. 查询关键词在元数据中的匹配
        # 检查查询词是否出现在标题或标题链中
        query_words = list(jieba.cut(query))
        for word in query_words:
            if len(word) < 2:
                continue
            if word in section_title:
                bonus += 0.02
            for heading in heading_chain:
                if word in heading:
                    bonus += 0.015
                    break
        
        # 4. 标题链长度加分（上下文信息丰富的文档）
        if len(heading_chain) >= 2:
            bonus += 0.01
        if len(heading_chain) >= 3:
            bonus += 0.005
        
        # 限制最大加分值
        return min(bonus, self._config.metadata_bonus_max)

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """混合检索：向量检索 + BM25 关键词检索 + RRF 融合 + 元数据增强。

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
        
        # 批量获取所有候选文档的元数据
        all_metadata = self._batch_get_metadata(all_ids)

        fused_docs: List[tuple] = []
        for doc_id in all_ids:
            vector_score = 1.0 / (self._vector_k + vector_rank.get(doc_id, candidate_count) + 1)
            bm25_score = 1.0 / (self._bm25_k + bm25_rank.get(doc_id, candidate_count) + 1)
            
            # 元数据增强：基于文档结构和语义特征提升相关性分数
            metadata = all_metadata.get(doc_id, {})
            score_bonus = self._calculate_metadata_bonus(query, metadata)

            fused_score = vector_score + bm25_score + score_bonus
            fused_docs.append((doc_id, fused_score))

        fused_docs.sort(key=lambda x: x[1], reverse=True)
        
        # 获取候选文档内容用于语义过滤
        all_doc_data = self._collection.get(ids=[doc[0] for doc in fused_docs], include=["documents"])
        id_to_doc = dict(zip(all_doc_data["ids"], all_doc_data["documents"]))
        
        # 语义相关性过滤：移除完全不相关的噪声文档
        filtered_docs = []
        for doc_id, score in fused_docs:
            doc_text = id_to_doc.get(doc_id, "")
            if self._check_semantic_relevance(query, doc_text):
                filtered_docs.append((doc_id, score))
        
        # 取 top_k 个相关文档
        sorted_ids = [doc[0] for doc in filtered_docs[:top_k]]
        
        # 记录元数据增强效果日志
        logger.debug("元数据增强效果 - 查询: %s", query[:50])
        for doc_id, score in fused_docs[:top_k]:
            meta = all_metadata.get(doc_id, {})
            logger.debug("  ID: %s, 分数: %.4f, 标题: %s", 
                        doc_id[:8], score, meta.get("section_title", ""))
        
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

    def _extract_core_keywords(self, query: str) -> List[str]:
        """
        从查询中提取核心关键词（长度>=2 的实词）。
        
        Args:
            query: 用户查询文本。
            
        Returns:
            核心关键词列表。
        """
        words = list(jieba.cut(query))
        stop_words = {"的", "了", "是", "在", "有", "和", "与", "或", "但", "而", "如果", "那么", "什么", "如何", "怎么", "为什么"}
        return [w for w in words if len(w) >= 2 and w not in stop_words]

    def _check_semantic_relevance(self, query: str, document_text: str, threshold: float = 0.3) -> bool:
        """
        检查文档与查询的语义相关性，过滤完全不相关的噪声文档。
        
        策略：
        1. 提取查询的核心关键词
        2. 检查核心关键词是否在文档中出现
        3. 如果至少有一个核心关键词匹配，则认为相关
        
        Args:
            query: 用户查询文本。
            document_text: 文档内容。
            threshold: 最低匹配比例要求（0.0-1.0）。
            
        Returns:
            如果文档相关返回 True，否则返回 False。
        """
        core_keywords = self._extract_core_keywords(query)
        
        if not core_keywords:
            return True
        
        doc_lower = document_text.lower()
        match_count = sum(1 for kw in core_keywords if kw in doc_lower)
        
        match_ratio = match_count / len(core_keywords)
        
        is_relevant = match_ratio >= threshold
        
        if not is_relevant:
            logger.debug(
                "语义过滤：查询 '%s' 的关键词 %s 在文档中仅匹配 %d/%d (比例: %.2f)，判定为噪声",
                query[:30], core_keywords, match_count, len(core_keywords), match_ratio
            )
        
        return is_relevant

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
