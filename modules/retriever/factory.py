from typing import Any, Dict

from modules.retriever.base import BaseRetriever
from modules.retriever.vector_retriever import VectorRetriever
from modules.retriever.hybrid_retriever import HybridRetriever
from modules.logger import get_logger

logger = get_logger(__name__)


class RetrieverFactory:
    """检索器工厂，根据类型名称创建对应的检索器实例。"""

    @staticmethod
    def create(retriever_type: str, **kwargs: Any) -> BaseRetriever:
        """
        Args:
            retriever_type: 检索器类型，"vector" 或 "hybrid"。
            **kwargs: 传递给对应检索器构造函数的参数。

        Returns:
            对应的 BaseRetriever 子类实例。

        Raises:
            ValueError: 当传入未知的 retriever_type 时。
        """
        registry: Dict[str, type] = {
            "vector": VectorRetriever,
            "hybrid": HybridRetriever,
        }
        if retriever_type not in registry:
            raise ValueError(
                f"未知的检索器类型: '{retriever_type}'，"
                f"可选值: {list(registry.keys())}"
            )
        logger.info("创建检索器: type=%s", retriever_type)
        return registry[retriever_type](**kwargs)
