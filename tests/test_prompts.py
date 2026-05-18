"""prompts 模块测试。"""

import pytest
from unittest.mock import patch

from modules.prompts import (
    PROMPT_TEMPLATES,
    get_prompt_template,
    build_prompt,
)


class TestGetPromptTemplate:
    """测试 get_prompt_template 函数。"""

    def test_get_prompt_template_strict(self):
        """验证返回 strict 模板且含有关键字。"""
        template = get_prompt_template("strict")
        assert isinstance(template, str)
        assert "我无法回答您的问题" in template
        assert "{contents}" in template
        assert "{user_query}" in template

    def test_get_prompt_template_balanced(self):
        """验证返回 balanced 模板且含有关键字。"""
        template = get_prompt_template("balanced")
        assert isinstance(template, str)
        assert "知识问答助手" in template
        assert "{contents}" in template
        assert "{user_query}" in template

    def test_get_prompt_template_creative(self):
        """验证返回 creative 模板且含有关键字。"""
        template = get_prompt_template("creative")
        assert isinstance(template, str)
        assert "创造力" in template
        assert "{contents}" in template
        assert "{user_query}" in template

    def test_get_prompt_template_unknown(self):
        """验证未知策略退回 strict 并记录警告。"""
        with patch("modules.prompts.logger") as mock_logger:
            template = get_prompt_template("nonexistent_strategy")
            mock_logger.warning.assert_called_once()
            assert "我无法回答您的问题" in template
            assert template == PROMPT_TEMPLATES["strict"]


class TestBuildPrompt:
    """测试 build_prompt 函数。"""

    def test_build_prompt_with_contents(self):
        """验证使用指定内容构建 Prompt 的结果。"""
        prompt = build_prompt(
            strategy="strict",
            contents="已知信息示例内容。",
            user_query="这是一个测试问题。",
        )
        assert "已知信息示例内容。" in prompt
        assert "这是一个测试问题。" in prompt
        assert "{contents}" not in prompt
        assert "{user_query}" not in prompt

    def test_build_prompt_empty_contents(self):
        """验证 contents 为空时使用'无相关已知信息'占位。"""
        prompt = build_prompt(
            strategy="strict",
            contents="",
            user_query="测试问题",
        )
        assert "无相关已知信息" in prompt

    def test_build_prompt_balanced(self):
        """验证 balanced 策略构建的 Prompt 正确填充。"""
        prompt = build_prompt(
            strategy="balanced",
            contents="参考文档内容。",
            user_query="帮我解答这个问题。",
        )
        assert "参考文档内容。" in prompt
        assert "帮我解答这个问题。" in prompt

    def test_build_prompt_creative(self):
        """验证 creative 策略构建的 Prompt 正确填充。"""
        prompt = build_prompt(
            strategy="creative",
            contents="相关资料。",
            user_query="说说你的看法。",
        )
        assert "相关资料。" in prompt
        assert "说说你的看法。" in prompt
