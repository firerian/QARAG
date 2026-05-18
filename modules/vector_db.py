import uuid
import hashlib
import datetime
from typing import List, Optional, Any, Dict

import chromadb
from chromadb.config import Settings
import jieba
from rank_bm25 import BM25Okapi
from modules.config import get_config
from modules.logger import get_logger
from modules.retriever.factory import RetrieverFactory

logger = get_logger(__name__)

class MyVectorDBConnector:
    # 初始化，传入集合名称，和向量化函数名
    def __init__(self, collection_name: str, client: Any, config: Optional[Any] = None) -> None:
        if config is None:
            config = get_config()
        self.config = config

        # 当前配置中，数据保存在内存中，如果需要持久化到磁盘，需使用 PersistentClient创建客户端
        # chroma_client = chromadb.Client()
        # 持久化到磁盘
        chroma_client = chromadb.PersistentClient(path=config.vector_db_path)

        # 创建一个 collection
        self.collection = chroma_client.get_or_create_collection(
            name=collection_name,
            metadata={'hnsw:space': 'cosine'}  # 推荐使用 cosine 余弦相似度
       )

        # 连接大模型的客户端
        self.client = client

        # 2. 初始化 BM25 (关键词检索)
        # 从 ChromaDB 中拉取所有已存在的文档，为 BM25 建立索引
        all_docs = self.collection.get(include=["documents"])["documents"]
        # 【新增】加一个判断：只有当数据库里有文档时，才初始化 BM25
        if all_docs:
            # 对中文进行分词处理
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            # 保存文档ID，用于 BM25 检索后匹配回原文
            self.doc_ids = self.collection.get(include=[])["ids"]
        else:
            # 如果数据库为空，先将 BM25 相关变量设为 None
            self.tokenized_corpus = []
            self.bm25 = None
            self.doc_ids = []

        self._init_retriever()

    def _init_retriever(self) -> None:
        """
        初始化内部检索器实例，供外部通过工厂模式复用检索逻辑。
        创建 VectorRetriever 和 HybridRetriever 并保存为实例属性。
        """
        self._vector_retriever = RetrieverFactory.create(
            "vector",
            collection=self.collection,
            embedding_client=self.client,
        )
        self._hybrid_retriever = RetrieverFactory.create(
            "hybrid",
            collection=self.collection,
            embedding_client=self.client,
            tokenized_corpus=self.tokenized_corpus,
            bm25=self.bm25,
            doc_ids=self.doc_ids,
            rrf_k=self.config.rrf_k,
        )

    # 批量向量化（使用 LangChain 封装好的方法，自动处理 batch）
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        # embed_documents 是 LangChain 的标准接口，直接传入文本列表即可
        return self.client.embed_documents(texts)

    # 添加文档与向量
    def add_documents(self, documents: List[str], instructions: Optional[List[str]] = None, dedup_strategy: Optional[str] = None) -> None:
        """
        向向量数据库添加文档。
        :param documents: 存入数据库的原始文本（如答案、段落内容）
        :param instructions: (可选) 用于生成向量的文本（如问题、指令）。如果不传，则默认用 documents 生成向量
        :param dedup_strategy: (可选) 去重策略，"skip" 跳过重复，"overwrite" 覆盖重复。默认使用 config.dedup_strategy
        """
        if dedup_strategy is None:
            dedup_strategy = self.config.dedup_strategy

        # 计算每个文档的 MD5 哈希
        doc_hashes = [hashlib.md5(doc.encode('utf-8')).hexdigest() for doc in documents]

        # 获取现有文档的哈希用于去重
        existing = self.collection.get(include=["metadatas"])
        existing_hashes = {}
        if existing["metadatas"]:
            for i, meta in enumerate(existing["metadatas"]):
                if meta and "md5" in meta:
                    existing_hashes[meta["md5"]] = existing["ids"][i]

        # 根据去重策略筛选文档
        valid_docs = []
        valid_hashes = []
        overwrite_count = 0
        skip_count = 0
        for i, (doc, doc_hash) in enumerate(zip(documents, doc_hashes)):
            if doc_hash in existing_hashes:
                if dedup_strategy == "overwrite":
                    self.collection.delete(ids=[existing_hashes[doc_hash]])
                    del existing_hashes[doc_hash]
                    valid_docs.append(doc)
                    valid_hashes.append(doc_hash)
                    overwrite_count += 1
                else:
                    logger.info(f"跳过重复文档 (MD5: {doc_hash[:8]}...)")
                    skip_count += 1
            else:
                valid_docs.append(doc)
                valid_hashes.append(doc_hash)

        if not valid_docs:
            logger.warning("所有文档均为重复内容，未添加任何文档")
            return

        if overwrite_count > 0:
            logger.info(f"覆盖 {overwrite_count} 条重复文档")
        if skip_count > 0:
            logger.info(f"跳过 {skip_count} 条重复文档")

        # 生成向量时使用 instructions 或 documents
        if instructions is not None:
            # 只对有效文档对应的 instructions 生成向量
            valid_indices = [i for i, doc_hash in enumerate(doc_hashes) if doc_hash in set(valid_hashes)]
            texts_to_embed = [instructions[i] for i in valid_indices]
        else:
            texts_to_embed = valid_docs

        embeddings = self.get_embeddings_batch(texts_to_embed)

        # 使用 UUID 生成唯一 ID
        ids = [str(uuid.uuid4()) for _ in range(len(valid_docs))]

        # 构建元数据
        metadatas = [
            {"md5": md5_hash, "timestamp": datetime.datetime.now().isoformat(), "source": "add_documents"}
            for md5_hash in valid_hashes
        ]

        self.collection.add(
            embeddings=embeddings,
            documents=valid_docs,
            ids=ids,
            metadatas=metadatas,
        )

        # 增量更新 BM25：只对新增文档进行分词并追加
        new_tokenized = [list(jieba.cut(doc)) for doc in valid_docs]
        self.tokenized_corpus.extend(new_tokenized)
        self.doc_ids.extend(ids)

        # 如果有覆盖删除，需要重建 BM25 以保证一致性
        if overwrite_count > 0:
            self.rebuild_bm25_index()
        elif new_tokenized:
            self.bm25 = BM25Okapi(self.tokenized_corpus)

        logger.info(f"成功存入 {len(valid_docs)} 条数据，当前库中共有: {self.collection.count()} 条")

    def rebuild_bm25_index(self) -> None:
        """
        从 ChromaDB 重新拉取全部文档，重建 BM25 索引。
        适用于索引损坏恢复或手动强制重建的场景。
        """
        all_docs = self.collection.get(include=["documents", "metadatas"])
        self.doc_ids = all_docs["ids"]
        if all_docs["documents"]:
            self.tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs["documents"]]
            self.bm25 = BM25Okapi(self.tokenized_corpus)
            logger.info(f"BM25 索引已重建，共 {len(self.tokenized_corpus)} 条文档")
        else:
            self.tokenized_corpus = []
            self.bm25 = None
            logger.warning("数据库为空，BM25 索引已清空")

    # 检索向量数据库

    def hybrid_search(self, query: str, top_k: int = 5) -> List[str]:
        # --- 1. 向量检索 ---
        query_embedding = self.client.embed_query(query)
        vector_results = self.collection.query(
            query_embeddings=[query_embedding], n_results=top_k * 3
        )
        # 【修改点1】ChromaDB返回的是二维列表，我们需要取第一个元素来“拍平”它
        vector_docs = vector_results['documents'][0] if vector_results['documents'] else []
        vector_ids = vector_results['ids'][0] if vector_results['ids'] else []

        # --- 2. BM25 关键词检索 ---
        bm25_top_ids = []
        if self.bm25 is not None:
            tokenized_query = list(jieba.cut(query))
            bm25_scores = self.bm25.get_scores(tokenized_query)
            id_score_pairs = list(zip(self.doc_ids, bm25_scores))
            id_score_pairs.sort(key=lambda x: x[1], reverse=True)  # 按分数排序
            bm25_top_ids = [pair[0] for pair in id_score_pairs[:top_k * 3]]  # 与向量检索同宽，避免关键片段被遗漏

        # --- 3. 结果融合 (RRF 倒数排名融合) ---
        fused_scores = {}
        k = self.config.rrf_k

        # 【修改点2】现在 vector_ids 已经是一维列表了，可以直接遍历
        for rank, doc_id in enumerate(vector_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        for rank, doc_id in enumerate(bm25_top_ids):
            fused_scores[doc_id] = fused_scores.get(doc_id, 0) + 1 / (k + rank + 1)

        # 按融合分数从高到低排序
        sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)

        # --- 4. 提取最终文档 ---
        final_docs = []
        if sorted_ids:
            # 从 ChromaDB 中根据排好序的 ID 提取出最终的文档内容
            all_data = self.collection.get(ids=sorted_ids[:top_k], include=["documents"])

            # 【核心修复】使用 zip 确保 ID 和 文档内容精准一一映射，防止顺序错乱
            id_to_doc = {doc_id: doc for doc_id, doc in zip(all_data["ids"], all_data["documents"])}

            # 按照 RRF 融合后的排名顺序提取文档
            final_docs = [id_to_doc[doc_id] for doc_id in sorted_ids[:top_k] if doc_id in id_to_doc]

        return final_docs


    def search(self, query: str, top_k: int) -> Dict[str, Any]:
        # 检索时，需要先把用户的提问转成向量
        query_embedding = self.client.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],  # 注意这里需要包一层列表
            n_results=top_k
        )
        return results

    # 查询存储在向量数据库的数据。仅限于测试，实际使用中，请勿使用。该方法会返回向量数据库中的所有数据，包括文档内容、向量、元数据和ID。

    def get(self, count: int) -> Dict[str, Any]:
        results = self.collection.get(include=["documents", "embeddings", "metadatas"], limit=count)
        return results