import json
import os
from typing import List, Tuple
from langchain_text_splitters import RecursiveCharacterTextSplitter


class DataProcessor:
    """
    负责加载、读取和切分原始文档数据的处理器
    """
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        初始化文本切分器
        :param chunk_size: 切分块的大小（字符数）
        :param chunk_overlap: 切分块之间的重叠字符数（保持上下文连贯）
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", "？", "！", "，", ""]
        )

    def load_qa_json(self, file_path: str) -> Tuple[List[str], List[str]]:
        """
        加载 JSON 格式的问答对数据 (如 train.json)
        :param file_path: 文件路径
        :return: (问题列表, 答案列表)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        instructions = []
        outputs = []

        # 逐行读取 JSON (JSONL 格式)
        with open(file_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    data = json.loads(line.strip())
                    instructions.append(data['instruction'])
                    outputs.append(data['output'])
                except Exception as e:
                    print(f"⚠️ 警告: 第 {line_num} 行数据解析失败，已跳过。错误: {e}")

        print(f"✅ 成功加载 {len(instructions)} 条问答对数据")
        return instructions, outputs

    def load_and_split_text(self, file_path: str) -> List[str]:
        """
        加载普通长文本文件 (如 .txt, .md) 并进行自动切分
        :param file_path: 文件路径
        :return: 切分后的文本片段列表
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"找不到文件: {file_path}")

        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()

        if not content.strip():
            print(f"⚠️ 警告: 文件 {file_path} 内容为空！")
            return []

        # 使用初始化时定义的切分器进行切分
        texts = self.text_splitter.split_text(content)
        print(f"✅ 成功加载并切分长文本，共切分为 {len(texts)} 个片段")
        return texts

    def split_text_content(self, content: str) -> List[str]:
        """
        直接对传入的文本字符串进行切分（适用于不需要读文件，直接传字符串的场景）
        :param content: 原始文本字符串
        :return: 切分后的文本片段列表
        """
        if not content.strip():
            return []
        texts = self.text_splitter.split_text(content)
        return texts