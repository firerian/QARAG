from typing import List, Any

import chromadb
from langchain_core.embeddings import Embeddings

from modules.retriever.base import BaseRetriever
from modules.logger import get_logger

logger = get_logger(__name__)


class VectorRetriever(BaseRetriever):
    """基于向量相似度的检索器，使用 ChromaDB 进行向量检索。"""

    def __init__(
        self,
        collection: chromadb.Collection,
        embedding_client: Embeddings,
    ) -> None:
        """
        Args:
            collection: ChromaDB 集合实例。
            embedding_client: LangChain embedding 客户端，用于生成查询向量。
        """
        self._collection = collection
        self._embedding_client = embedding_client

    def search(self, query: str, top_k: int = 5) -> List[str]:
        """向量检索：将查询文本转向量后在 ChromaDB 中检索最相似文档。

        Args:
            query: 用户查询文本。
            top_k: 返回的文档数量。

        Returns:
            按相似度排序的文档文本列表。
        """
        query_embedding = self._embedding_client.embed_query(query)
        results = self._collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
        )
        documents: List[Any] = results.get("documents", [])
        if documents and documents[0]:
            return list(documents[0])
        logger.warning("向量检索未找到匹配文档，query=%s", query[:50])
        return []
