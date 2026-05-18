"""DataProcessor 测试模块。"""

import json
import os
import tempfile
import pytest

from modules.data_processor import DataProcessor


class TestLoadQAJson:
    """测试 load_qa_json 方法。"""

    def test_load_qa_json_success(self):
        """创建一个临时 JSONL 文件，加载并验证结果。"""
        data = [
            {"instruction": "问题一", "output": "答案一"},
            {"instruction": "问题二", "output": "答案二"},
            {"instruction": "问题三", "output": "答案三"},
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
            temp_path = f.name

        try:
            processor = DataProcessor()
            instructions, outputs = processor.load_qa_json(temp_path)
            assert len(instructions) == 3
            assert len(outputs) == 3
            assert instructions == ["问题一", "问题二", "问题三"]
            assert outputs == ["答案一", "答案二", "答案三"]
        finally:
            os.unlink(temp_path)

    def test_load_qa_json_file_not_found(self):
        """验证文件不存在时抛出 FileNotFoundError。"""
        processor = DataProcessor()
        with pytest.raises(FileNotFoundError, match="找不到文件"):
            processor.load_qa_json("/nonexistent/path/file.json")

    def test_load_qa_json_corrupted_line(self):
        """验证损坏的行被跳过，正常行仍被加载。"""
        lines = [
            '{"instruction": "问题一", "output": "答案一"}\n',
            '这不是有效的 JSON 行\n',
            '{"instruction": "问题二", "output": "答案二"}\n',
        ]
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", encoding="utf-8", delete=False
        ) as f:
            f.writelines(lines)
            temp_path = f.name

        try:
            processor = DataProcessor()
            instructions, outputs = processor.load_qa_json(temp_path)
            assert len(instructions) == 2
            assert len(outputs) == 2
            assert "问题一" in instructions
            assert "问题二" in instructions
        finally:
            os.unlink(temp_path)


class TestLoadAndSplitText:
    """测试 load_and_split_text 方法。"""

    def test_load_and_split_text_success(self):
        """创建临时文本文件，加载并切分后验证结果。"""
        content = "这是第一段内容。\n\n这是第二段内容。\n\n这是第三段内容。"
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write(content)
            temp_path = f.name

        try:
            processor = DataProcessor()
            chunks = processor.load_and_split_text(temp_path)
            assert isinstance(chunks, list)
            assert len(chunks) > 0
            for chunk in chunks:
                assert isinstance(chunk, str)
                assert len(chunk) > 0
        finally:
            os.unlink(temp_path)

    def test_load_and_split_text_file_not_found(self):
        """验证文件不存在时抛出 FileNotFoundError。"""
        processor = DataProcessor()
        with pytest.raises(FileNotFoundError, match="找不到文件"):
            processor.load_and_split_text("/nonexistent/path/file.txt")

    def test_load_and_split_text_empty_file(self):
        """验证空文件返回空列表。"""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", encoding="utf-8", delete=False
        ) as f:
            f.write("")
            temp_path = f.name

        try:
            processor = DataProcessor()
            chunks = processor.load_and_split_text(temp_path)
            assert chunks == []
        finally:
            os.unlink(temp_path)


class TestSplitTextContent:
    """测试 split_text_content 方法。"""

    def test_split_text_content(self):
        """验证直接传入字符串进行切分。"""
        content = "这是第一部分内容。这是第二部分内容。这是第三部分内容。"
        processor = DataProcessor(chunk_size=20, chunk_overlap=0)
        chunks = processor.split_text_content(content)
        assert isinstance(chunks, list)
        assert len(chunks) > 0
        for chunk in chunks:
            assert isinstance(chunk, str)
            assert len(chunk) > 0

    def test_split_text_content_empty(self):
        """验证空字符串返回空列表。"""
        processor = DataProcessor()
        chunks = processor.split_text_content("   ")
        assert chunks == []


class TestChineseSeparators:
    """测试中文标点符号切分。"""

    def test_chinese_separators(self):
        """验证使用中文标点（。？！，）作为分隔符能正确切分文本。"""
        content = (
            "今天天气很好。我们出去玩吧！你说怎么样？"
            "那好吧，我们走。"
        )
        processor = DataProcessor(chunk_size=10, chunk_overlap=0)
        chunks = processor.split_text_content(content)
        assert isinstance(chunks, list)
        assert len(chunks) >= 1
        for chunk in chunks:
            assert isinstance(chunk, str)
