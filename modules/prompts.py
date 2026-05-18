"""
可配置的 Prompt 模板管理模块。

提供三种预设策略的 Prompt 模板（strict / balanced / creative），
以及对模板进行获取、填充的工具函数。
"""

import logging
from typing import Dict

from modules.logger import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATES: Dict[str, str] = {
    "strict": (
        "你是一个严格的问答机器人。请严格遵守以下规则：\n"
        " 1. 严禁使用你自身的训练数据、常识或外部知识进行回答,"
        "不要自行总结，严格按照【已知信息】来回答问题。\n"
        " 2. 如果【已知信息】中没有直接包含回答用户问题所需的内容，"
        "你必须且只能回复：\"我无法回答您的问题\"。\n"
        " 3. 即使你知道答案，但只要【已知信息】里没有，就视为不知道。\n"
        " 【已知信息】\n"
        " {contents}\n"
        " ----\n"
        " 用户问：\n"
        " {user_query}\n"
        " 请用中文回答用户问题。\n"
    ),
    "balanced": (
        "你是一个知识问答助手。请根据以下规则回答问题：\n"
        " 1. 优先基于【已知信息】进行回答。\n"
        " 2. 如果【已知信息】不足以回答问题，"
        "你可以结合常识进行补充，但需要明确指出哪些来自已知信息，哪些来自你的常识。\n"
        " 3. 如果完全无法回答，请诚实地说\"抱歉，我无法回答这个问题\"。\n"
        " 【已知信息】\n"
        " {contents}\n"
        " ----\n"
        " 用户问：\n"
        " {user_query}\n"
        " 请用中文回答用户问题。\n"
    ),
    "creative": (
        "你是一个博学且富有创造力的知识问答助手。请参考以下信息回答用户问题：\n"
        "你可以自由运用你的知识和创造力来提供最全面、最有见地的回答。\n"
        "如果参考信息中有相关内容，请优先采纳；"
        "如果没有，请基于你的知识给出最佳答案。\n"
        " 【参考信息】\n"
        " {contents}\n"
        " ----\n"
        " 用户问：\n"
        " {user_query}\n"
        " 请用中文回答用户问题。\n"
    ),
}


def get_prompt_template(strategy: str = "strict") -> str:
    """
    根据策略名称获取对应的 Prompt 模板字符串。

    Args:
        strategy: 策略名称，支持 "strict"、"balanced"、"creative"。

    Returns:
        对应的模板字符串。若策略不存在，退回 "strict" 并记录警告。
    """
    if strategy in PROMPT_TEMPLATES:
        return PROMPT_TEMPLATES[strategy]

    logger.warning(
        "未知的 Prompt 策略 '%s'，已退回默认策略 'strict'", strategy
    )
    return PROMPT_TEMPLATES["strict"]


def build_prompt(strategy: str, contents: str, user_query: str) -> str:
    """
    使用指定策略构建最终的 Prompt 字符串。

    Args:
        strategy: 策略名称。
        contents: 检索到的已知信息文本，为空时自动替换为 "无相关已知信息"。
        user_query: 用户的原始问题。

    Returns:
        填充后的完整 Prompt 字符串。
    """
    template = get_prompt_template(strategy)
    display_contents = contents if contents else "无相关已知信息"
    return template.format(contents=display_contents, user_query=user_query)
