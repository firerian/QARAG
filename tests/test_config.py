"""Config 类测试模块。"""

import os
import pytest
from unittest.mock import patch

import modules.config as config_module


class TestConfigCreationWithDefaults:
    """测试 Config 的默认值是否正确。"""

    def test_config_creation_with_defaults(self, monkeypatch):
        """验证 Config 在设置必填字段后创建成功，且默认值正确。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-test-key")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com")

        from modules.config import Config
        cfg = Config()

        assert cfg.llm_api_key == "sk-test-key"
        assert cfg.llm_model == "gpt-4o"
        assert cfg.llm_base_url == "https://api.example.com"
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 2048
        assert cfg.vector_db_path == "./chroma_data"
        assert cfg.embedding_model == "bge-m3:567m"
        assert cfg.chunk_size == 500
        assert cfg.chunk_overlap == 50
        assert cfg.top_k == 5
        assert cfg.rrf_k == 30
        assert cfg.log_level == "INFO"
        assert cfg.dedup_strategy == "skip"
        assert cfg.prompt_strategy == "strict"


class TestConfigValidation:
    """测试 Config.validate() 的校验逻辑。"""

    def test_config_validate_missing_required(self, monkeypatch):
        """验证缺少 LLM_API_KEY 时抛出 ValueError。"""
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com")
        monkeypatch.delenv("LLM_API_KEY", raising=False)

        from modules.config import Config
        cfg = Config()
        with pytest.raises(ValueError, match="LLM_API_KEY"):
            cfg.validate()

    def test_config_validate_all_missing(self, monkeypatch):
        """验证所有必填字段均缺失时抛出 ValueError 并列出全部缺失项。"""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        from modules.config import Config
        cfg = Config()
        with pytest.raises(ValueError) as exc_info:
            cfg.validate()
        error_msg = str(exc_info.value)
        assert "LLM_API_KEY" in error_msg
        assert "LLM_MODEL" in error_msg
        assert "LLM_BASE_URL" in error_msg

    def test_config_validate_success(self, monkeypatch):
        """验证全部必填字段存在时不抛出异常。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-test")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com")

        from modules.config import Config
        cfg = Config()
        cfg.validate()


class TestConfigFromEnv:
    """测试 Config 从环境变量加载值。"""

    def test_config_from_env(self, monkeypatch):
        """验证自定义环境变量被正确读取到 Config 属性中。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-from-env")
        monkeypatch.setenv("LLM_MODEL", "custom-model")
        monkeypatch.setenv("LLM_BASE_URL", "https://custom.api.com")
        monkeypatch.setenv("TEMPERATURE", "0.7")
        monkeypatch.setenv("MAX_TOKENS", "4096")
        monkeypatch.setenv("VECTOR_DB_PATH", "/custom/path")
        monkeypatch.setenv("CHUNK_SIZE", "800")
        monkeypatch.setenv("CHUNK_OVERLAP", "100")
        monkeypatch.setenv("TOP_K", "10")
        monkeypatch.setenv("RRF_K", "30")
        monkeypatch.setenv("LOG_LEVEL", "DEBUG")
        monkeypatch.setenv("DEDUP_STRATEGY", "overwrite")
        monkeypatch.setenv("PROMPT_STRATEGY", "creative")

        from modules.config import Config
        cfg = Config()

        assert cfg.llm_api_key == "sk-from-env"
        assert cfg.llm_model == "custom-model"
        assert cfg.llm_base_url == "https://custom.api.com"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 4096
        assert cfg.vector_db_path == "/custom/path"
        assert cfg.chunk_size == 800
        assert cfg.chunk_overlap == 100
        assert cfg.top_k == 10
        assert cfg.rrf_k == 30
        assert cfg.log_level == "DEBUG"
        assert cfg.dedup_strategy == "overwrite"
        assert cfg.prompt_strategy == "creative"


class TestGetConfigSingleton:
    """测试 get_config() 单例行为。"""

    def test_get_config_singleton(self, monkeypatch):
        """验证 get_config 多次调用返回同一个实例。"""
        monkeypatch.setenv("LLM_API_KEY", "sk-singleton")
        monkeypatch.setenv("LLM_MODEL", "gpt-4o")
        monkeypatch.setenv("LLM_BASE_URL", "https://api.example.com")

        import modules.config as cfg_mod
        cfg_mod._config = None

        cfg1 = cfg_mod.get_config()
        cfg2 = cfg_mod.get_config()
        assert cfg1 is cfg2

    def test_get_config_validates(self, monkeypatch):
        """验证 get_config 在缺失必填项时抛出 ValueError。"""
        monkeypatch.delenv("LLM_API_KEY", raising=False)
        monkeypatch.delenv("LLM_MODEL", raising=False)
        monkeypatch.delenv("LLM_BASE_URL", raising=False)

        import modules.config as cfg_mod
        cfg_mod._config = None

        with pytest.raises(ValueError):
            cfg_mod.get_config()
