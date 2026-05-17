import os
import chromadb
from chromadb.config import Settings
import jieba
from rank_bm25 import BM25Okapi
from dotenv import load_dotenv
load_dotenv()



chroma_path = os.environ["VECTOR_DB_PATH"]
class MyVectorDBConnector:
    # 初始化，传入集合名称，和向量化函数名
    def __init__(self, collection_name,client):
        # 当前配置中，数据保存在内存中，如果需要持久化到磁盘，需使用 PersistentClient创建客户端
        # chroma_client = chromadb.Client()
        # 持久化到磁盘
        chroma_client = chromadb.PersistentClient(path=chroma_path)

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

    # 批量向量化（使用 LangChain 封装好的方法，自动处理 batch）
    def get_embeddings_batch(self, texts):
        # embed_documents 是 LangChain 的标准接口，直接传入文本列表即可
        return self.client.embed_documents(texts)

    # 添加文档与向量
    def add_documents(self, documents, instructions=None):
        """
        向向量数据库添加文档。
        :param documents: 存入数据库的原始文本（如答案、段落内容）
        :param instructions: (可选) 用于生成向量的文本（如问题、指令）。如果不传，则默认用 documents 生成向量
        """
        # 如果用户没有传入 instructions，就默认使用 documents 来进行向量化
        texts_to_embed = instructions if instructions is not None else documents

        # 使用确定的文本去生成向量
        embeddings = self.get_embeddings_batch(texts_to_embed)

        self.collection.add(
            embeddings=embeddings,
            documents=documents,
            ids=[f"id{i}" for i in range(len(documents))],
        )
        # 同步更新 BM25 的语料库（简单起见，这里每次添加都重新拉取全量建立索引）
        # 在生产环境中，建议增量更新 BM25 语料库
        all_docs = self.collection.get(include=["documents"])["documents"]
        self.tokenized_corpus = [list(jieba.cut(doc)) for doc in all_docs]
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        self.doc_ids = self.collection.get(include=[])["ids"]



        print(f"成功存入 {len(documents)} 条数据，当前库中共有: {self.collection.count()} 条")

    # 检索向量数据库

    def hybrid_search(self, query, top_k=5):
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
            bm25_top_ids = [pair[0] for pair in id_score_pairs[:top_k * 2]]  # 只取ID

        # --- 3. 结果融合 (RRF 倒数排名融合) ---
        fused_scores = {}
        k = 60

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


    def search(self, query, top_k):
        # 检索时，需要先把用户的提问转成向量
        query_embedding = self.client.embed_query(query)

        results = self.collection.query(
            query_embeddings=[query_embedding],  # 注意这里需要包一层列表
            n_results=top_k
        )
        return results

    # 查询存储在向量数据库的数据。仅限于测试，实际使用中，请勿使用。该方法会返回向量数据库中的所有数据，包括文档内容、向量、元数据和ID。

    def get(self, count):
        results = self.collection.get(include=["documents", "embeddings", "metadatas"], limit=count)
        return results