"""logger 模块测试。"""

import os
import logging
import pytest
import tempfile
from unittest.mock import patch

import modules.logger as logger_module


@pytest.fixture(autouse=True)
def clear_logger_cache():
    """每个测试前清除 logger 缓存，确保测试隔离。"""
    logger_module._LOG_CACHE.clear()
    yield
    logger_module._LOG_CACHE.clear()


class TestSetupLogger:
    """测试 setup_logger 函数。"""

    def test_setup_logger_creates_logs_dir(self):
        """验证 setup_logger 会创建 logs 目录。"""
        with tempfile.TemporaryDirectory() as tmpdir:
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.object(logger_module, "LOG_FORMAT", logger_module.LOG_FORMAT):
                    test_logger = logger_module.setup_logger("test_setup", log_level="DEBUG")
                assert os.path.isdir("logs")
            finally:
                for handler in test_logger.handlers[:]:
                    handler.close()
                    test_logger.removeHandler(handler)
                os.chdir(original_cwd)

    def test_get_logger_returns_logger(self, monkeypatch):
        """验证 get_logger 返回 Logger 实例。"""
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        logger_inst = logger_module.get_logger("test_logger_instance")
        assert isinstance(logger_inst, logging.Logger)

    def test_get_logger_cache(self):
        """验证相同名称返回相同的 logger 实例（缓存生效）。"""
        logger_a = logger_module.get_logger("cached_name")
        logger_b = logger_module.get_logger("cached_name")
        assert logger_a is logger_b

    def test_get_logger_different_names(self):
        """验证不同名称返回不同的 logger 实例。"""
        logger_a = logger_module.get_logger("name_a")
        logger_b = logger_module.get_logger("name_b")
        assert logger_a is not logger_b


class TestLoggerLevels:
    """测试日志级别。"""

    def test_logger_levels(self, monkeypatch):
        """验证不同日志级别配置生效。"""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR"]:
            monkeypatch.setenv("LOG_LEVEL", level)
            logger_module._LOG_CACHE.clear()
            logger_inst = logger_module.get_logger(f"test_level_{level}")
            log_level_num = getattr(logging, level)
            assert logger_inst.getEffectiveLevel() <= log_level_num
