import os
import logging
from typing import Dict


_LOG_CACHE: Dict[str, logging.Logger] = {}

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


def setup_logger(name: str, log_level: str = "INFO") -> logging.Logger:
    """
    配置并返回一个同时输出到控制台和文件的 logger 实例。

    Args:
        name: logger 的名称。
        log_level: 控制台输出的日志级别，默认 "INFO"。

    Returns:
        配置完成的 logging.Logger 实例。
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    formatter = logging.Formatter(LOG_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        os.path.join(log_dir, "rag.log"), encoding="utf-8"
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "rag") -> logging.Logger:
    """
    获取指定名称的 logger 实例，支持缓存以避免重复配置。

    日志级别从环境变量 LOG_LEVEL 读取，默认 "INFO"。

    Args:
        name: logger 的名称，默认 "rag"。

    Returns:
        配置完成的 logging.Logger 实例。
    """
    if name in _LOG_CACHE:
        return _LOG_CACHE[name]

    log_level = os.getenv("LOG_LEVEL", "INFO")
    logger = setup_logger(name, log_level)
    _LOG_CACHE[name] = logger

    return logger
