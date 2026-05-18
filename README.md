<<<<<<< HEAD

# QA-RAG 智能问答系统

基于检索增强生成（RAG）技术的智能问答系统，支持向量检索与 BM25 关键词检索的混合搜索模式，提供灵活可配置的 Prompt 策略，可快速构建领域专属的知识库问答应用。

## 核心功能

- **混合检索**：融合向量语义检索与 BM25 关键词检索，通过 RRF（倒数排名融合）算法实现最优排序
- **多策略 Prompt**：内置 strict（严格）、balanced（平衡）、creative（创意）三种 Prompt 策略，适配不同业务场景
- **智能去重**：支持基于 MD5 的文档去重，可选择跳过或覆盖策略
- **灵活数据导入**：支持 JSONL 问答对数据和普通文本文件的批量导入与自动切分
- **持久化存储**：基于 ChromaDB 实现向量数据持久化，支持增量更新和索引重建
- **可配置化**：全部参数通过环境变量配置，开箱即用

## 技术栈

| 技术               | 用途                   |
| ---------------- | -------------------- |
| Python 3.10+     | 核心运行环境               |
| LangChain        | LLM 和 Embedding 统一接口 |
| ChromaDB         | 向量数据库，持久化存储          |
| Ollama           | 本地 Embedding 模型服务    |
| BM25 (rank-bm25) | 关键词检索算法              |
| Jieba            | 中文分词工具               |
| OpenAI SDK       | 兼容 OpenAI 接口的 LLM    |

## 安装与配置

### 环境要求

- Python 3.10 或更高版本
- Ollama（用于本地 Embedding 模型）

### 1. 安装依赖

```bash
pip install -r  requirements.txt
```

### 2. 安装并启动 Ollama

1. 从 [Ollama 官网](https://ollama.com/) 下载并安装
2. 启动 Ollama 服务
3. 下载 Embedding 模型：

```bash
ollama pull bge-m3:567m
```

> 也可使用其他兼容 Ollama 接口的 Embedding 模型，修改 `EMBEDDING_MODEL` 环境变量即可。

### 3. 配置环境变量

在项目根目录创建 `.env` 文件，填写以下配置：

```env
# LLM 配置（必填）
LLM_API_KEY=your_api_key_here
LLM_MODEL=your_model_name
LLM_BASE_URL=https://your-api-endpoint.com/v1

# Embedding 配置
EMBEDDING_MODEL=bge-m3:567m

# 向量数据库路径
VECTOR_DB_PATH=./chroma_data

# 文本切分配置
CHUNK_SIZE=500
CHUNK_OVERLAP=50

# 检索配置
TOP_K=5
RRF_K=60

# LLM 生成配置
TEMPERATURE=0.0
MAX_TOKENS=2048

# 日志级别
LOG_LEVEL=INFO

# 去重策略: skip（跳过）或 overwrite（覆盖）
DEDUP_STRATEGY=skip

# Prompt 策略: strict / balanced / creative
PROMPT_STRATEGY=strict
```

| 环境变量              | 说明                  | 默认值             | 必填 |
| ----------------- | ------------------- | --------------- | -- |
| `LLM_API_KEY`     | LLM API 密钥          | 无               | 是  |
| `LLM_MODEL`       | 使用的模型名称             | 无               | 是  |
| `LLM_BASE_URL`    | LLM API 地址          | 无               | 是  |
| `EMBEDDING_MODEL` | Ollama Embedding 模型 | `bge-m3:567m`   | 否  |
| `VECTOR_DB_PATH`  | 向量数据持久化路径           | `./chroma_data` | 否  |
| `CHUNK_SIZE`      | 文本切分块大小             | `500`           | 否  |
| `CHUNK_OVERLAP`   | 切分块重叠字符数            | `50`            | 否  |
| `TOP_K`           | 检索返回的文档数量           | `5`             | 否  |
| `RRF_K`           | RRF 融合算法超参数 k       | `60`            | 否  |
| `TEMPERATURE`     | LLM 生成温度            | `0.0`           | 否  |
| `MAX_TOKENS`      | LLM 最大生成 token 数    | `2048`          | 否  |
| `LOG_LEVEL`       | 日志级别                | `INFO`          | 否  |
| `DEDUP_STRATEGY`  | 文档去重策略              | `skip`          | 否  |
| `PROMPT_STRATEGY` | Prompt 策略           | `strict`        | 否  |

## 命令使用指南

### CLI 参数说明

| 参数                  | 简写   | 类型                   | 默认值      | 说明               |
| ------------------- | ---- | -------------------- | -------- | ---------------- |
| `--retriever`       | `-r` | `default` / `hybrid` | `hybrid` | 检索器类型            |
| `--query`           | `-q` | string               | 无        | 单次提问             |
| `--interactive`     | `-i` | flag                 | `false`  | 进入交互问答模式         |
| `--ingest-qa`       | -    | string               | 无        | 导入 JSONL 格式问答对文件 |
| `--ingest-text`     | -    | string               | 无        | 导入并切分普通文本文件      |
| `--rebuild`         | -    | flag                 | `false`  | 重建 BM25 索引       |
| `--log-level`       | -    | string               | 无        | 覆盖日志级别           |
| `--collection`      | `-c` | string               | `demo`   | 向量库集合名称          |
| `--prompt-strategy` | -    | string               | `strict` | Prompt 策略        |

### 数据导入

导入 JSONL 格式的问答对数据：

```bash
python main.py --ingest-qa ./data/qa_pairs.jsonl
```

导入普通文本文件并自动切分：

```bash
python main.py --ingest-text ./data/knowledge_base.txt
```

指定自定义集合名称：

```bash
python main.py --ingest-qa ./data/qa.jsonl -c my_collection
```

### 单次提问

使用混合检索器和 strict 策略：

```bash
python main.py -q "什么是RAG技术？"
```

指定检索器和 Prompt 策略：

```bash
python main.py -q "什么是RAG技术？" -r default --prompt-strategy balanced
```

### 交互问答模式

进入交互式对话：

```bash
python main.py -i
```

指定检索器和策略：

```bash
python main.py -i -r hybrid --prompt-strategy creative
```

在交互模式下，输入 `quit` 或 `exit` 可退出对话。

### 索引重建

强制重建 BM25 索引（适用于索引损坏或数据不一致时）：

```bash
python main.py --rebuild
```

结合数据导入一起执行：

```bash
python main.py --ingest-text ./data/new_doc.txt --rebuild
```

### 日志调试

```bash
python main.py -q "测试问题" --log-level DEBUG
```

## 数据生成方法

### JSONL 问答对数据

系统支持逐行读取 JSON（JSONL）格式的问答对数据。每行一个 JSON 对象，包含 `instruction`（问题）和 `output`（答案）字段。

示例文件 `qa_pairs.jsonl`：

```jsonl
{"instruction": "什么是零信任安全架构？", "output": "零信任安全架构（Zero Trust Architecture）是一种安全理念，其核心思想是默认不信任任何内部或外部的用户、设备或应用，必须进行严格的身份验证和授权。"}
{"instruction": "如何防范 SQL 注入攻击？", "output": "防范 SQL 注入的主要方法包括：1. 使用参数化查询或预编译语句；2. 对用户输入进行严格的验证和过滤；3. 使用 ORM 框架；4. 最小化数据库权限。"}
{"instruction": "什么是 RAG 技术？", "output": "RAG（Retrieval-Augmented Generation）即检索增强生成，是一种将信息检索与语言模型生成相结合的技术。它先从知识库中检索相关文档，再将检索结果与用户问题一起输入给 LLM，从而生成更准确、更有依据的回答。"}
```

> 解析失败的数据行会被自动跳过并输出警告，不影响整体导入流程。

### 普通文本文件

支持 `.txt`、`.md` 等纯文本格式。系统会使用 `RecursiveCharacterTextSplitter` 自动进行递归字符切分。

切分优先级（分隔符顺序）：`\n\n` → `\n` → `。` → `？` → `！` → `，` → 字符级别

### 数据切分策略

| 参数              | 说明            | 建议                          |
| --------------- | ------------- | --------------------------- |
| `CHUNK_SIZE`    | 每个切分块的最大字符数   | 短文/问答对：200-500；长文档：500-1000 |
| `CHUNK_OVERLAP` | 相邻切分块之间的重叠字符数 | 一般为 CHUNK\_SIZE 的 10%-20%   |

较小的 `CHUNK_SIZE` 提高检索精度，较大的 `CHUNK_SIZE` 保留更多上下文。

### 数据导入流程

```
数据文件 → DataProcessor 加载 → 文本切分 → MyVectorDBConnector 向量化 → ChromaDB 存储
```

1. 使用 `--ingest-qa` 或 `--ingest-text` 指定数据文件
2. 系统自动进行文本切分（如使用 `--ingest-text`）
3. 生成向量并存储到 ChromaDB
4. 自动更新 BM25 关键词索引

## 使用技巧

### Prompt 策略选择

| 策略         | 行为                      | 适用场景               |
| ---------- | ----------------------- | ------------------ |
| `strict`   | 仅依据已知信息回答，信息不足时回复"无法回答" | 法律、医疗、安全等需要高准确性的场景 |
| `balanced` | 优先使用已知信息，不足时可补充常识并标注    | 一般知识问答、内部咨询        |
| `creative` | 自由运用知识，参考信息仅作参考         | 头脑风暴、创意分析、开放性讨论    |

使用示例：

```bash
python main.py -q "分析公司安全策略" --prompt-strategy balanced
```

### 性能优化建议

1. **合理设置 TOP\_K**：增大 `TOP_K` 可提高回答质量，但会增加上下文长度和 token 消耗
2. **调整 RRF\_K**：默认 60，较小的值让排名更敏感，较大的值让排名更平滑
3. **批量导入**：一次性导入更多数据，减少多次调用的开销
4. **使用 hybrid 检索**：混合检索在中文场景下效果优于纯向量检索
5. **定期重建索引**：在大量增删数据后，使用 `--rebuild` 确保 BM25 索引与向量数据一致

### 常见问题排查（FAQ）

**Q: 启动时报错 "Embedding 服务不可用"**
A: 确保 Ollama 服务已启动，并且指定的 Embedding 模型已下载。运行 `ollama list` 确认模型存在。

**Q: 导入数据后查询返回空结果**
A: 检查 `TOP_K` 参数是否过小，或尝试使用 `--log-level DEBUG` 查看详细检索过程。也可以使用 `--rebuild` 重建 BM25 索引。

**Q: LLM 回答与已知信息不符**
A: 使用 `strict` Prompt 策略，该策略会严格限制 LLM 仅基于检索到的信息回答。检查 `--prompt-strategy strict`。

**Q: 向量库占用空间过大**
A: 检查 `VECTOR_DB_PATH` 指向的目录，可手动清理 `./chroma_data` 目录后重新导入。

**Q: 重复数据多次导入**
A: 系统默认启用去重策略（`DEDUP_STRATEGY=skip`），重复文档会被自动跳过。如需更新内容，设置 `DEDUP_STRATEGY=overwrite`。

## 典型使用场景

### 场景一：企业安全培训知识库

**需求**：企业需要员工掌握信息安全规章制度，并提供随时可查询的问答服务。

**方案**：

1. 将企业安全手册、规章制度整理为 JSONL 问答对格式
2. 导入系统：`python main.py --ingest-qa ./data/security_qa.jsonl`
3. 使用 `strict` 策略确保回答的准确性：`python main.py -i --prompt-strategy strict`
4. 员工通过交互模式随时提问查询

**效果**：新员工可自助查询安全规定，降低培训成本，确保合规知识传递的准确性。

### 场景二：产品客服问答系统

**需求**：为产品建立智能客服，减少人工客服压力。

**方案**：

1. 将产品 FAQ、操作手册整理为 JSONL 格式
2. 导入数据：`python main.py --ingest-qa ./data/product_faq.jsonl`
3. 使用 `balanced` 策略，允许在知识库不足时适当补充：`python main.py -i --prompt-strategy balanced`
4. 集成到网站或应用的客服入口

**效果**：自动处理常见问题，人工客服只需处理复杂个案，大幅提升服务效率。

### 场景三：学术研究辅助

**需求**：研究者需要快速检索和问答特定领域的学术文献。

**方案**：

1. 将领域文献摘要、关键结论整理为文本文件
2. 导入并自动切分：`python main.py --ingest-text ./data/research_papers.txt`
3. 使用 `creative` 策略进行开放性分析：`python main.py -i --prompt-strategy creative`
4. 针对不同子领域使用不同的 collection 管理

**效果**：快速检索文献要点，辅助综述撰写和研究方向探索。

## 项目结构

```
QARAG/
├── main.py                 # 主入口，CLI 参数解析和流程调度
├── modules/
│   ├── __init__.py         # 模块导出
│   ├── config.py           # 全局配置管理（环境变量加载）
│   ├── data_processor.py   # 数据加载与文本切分
│   ├── llm_handler.py      # LLM 调用与 Embedding 管理
│   ├── vector_db.py        # ChromaDB 向量数据库封装
│   ├── prompts.py          # Prompt 模板管理
│   ├── logger.py           # 日志工具
│   ├── performance.py      # 性能监控
│   └── retriever/          # 检索器模块
│       ├── __init__.py
│       ├── base.py         # 检索器基类
│       ├── factory.py      # 检索器工厂
│       ├── vector_retriever.py   # 向量检索器
│       └── hybrid_retriever.py   # 混合检索器（向量 + BM25 + RRF）
└── tests/                  # 测试目录
    ├── conftest.py         # 测试共享 fixtures
    ├── test_config.py      # 配置测试
    ├── test_data_processor.py  # 数据处理测试
    ├── test_integration.py # 集成测试
    └── ...
```

## 测试

运行全部测试：

```bash
pytest
```

运行特定测试模块：

```bash
pytest tests/test_integration.py -v
```

运行带覆盖率报告的测试：

```bash
pytest --cov=modules --cov-report=term-missing
```

## 许可证

# 本项目仅供学习和研究使用。

# QARAG

基于 RAG 架构的智能问答系统 —— 混合向量/关键词检索 + 可配置 Prompt 策略 + 插件化检索引擎

> > > > > > > 574106f2c070c3b3310409dab4182bc6b1d5f7b3

