from abc import ABC, abstractmethod
from typing import List


class BaseRetriever(ABC):
    """检索器抽象基类，定义统一的检索接口。"""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> List[str]:
        """检索与查询最相关的文档列表。

        Args:
            query: 用户查询文本。
            top_k: 返回的文档数量。

        Returns:
            按相关度排序的文档文本列表。
        """
        pass
