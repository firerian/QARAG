import os
from dataclasses import dataclass, field
from typing import Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class Config:
    """
    应用全局配置类，所有配置项均从环境变量中加载。

    通过 get_config() 获取单例实例，确保整个应用使用同一份配置。
    """
    llm_api_key: str = field(
        default_factory=lambda: os.getenv("LLM_API_KEY")
    )
    llm_model: str = field(
        default_factory=lambda: os.getenv("LLM_MODEL")
    )
    llm_base_url: str = field(
        default_factory=lambda: os.getenv("LLM_BASE_URL")
    )
    temperature: float = field(
        default_factory=lambda: float(os.getenv("TEMPERATURE", "0.0"))
    )
    max_tokens: int = field(
        default_factory=lambda: int(os.getenv("MAX_TOKENS", "2048"))
    )
    vector_db_path: str = field(
        default_factory=lambda: os.getenv("VECTOR_DB_PATH", "./chroma_data")
    )
    embedding_model: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "bge-m3:567m")
    )
    chunk_size: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_SIZE", "500"))
    )
    chunk_overlap: int = field(
        default_factory=lambda: int(os.getenv("CHUNK_OVERLAP", "50"))
    )
    top_k: int = field(
        default_factory=lambda: int(os.getenv("TOP_K", "5"))
    )
    rrf_k: int = field(
        default_factory=lambda: int(os.getenv("RRF_K", "30"))
    )
    log_level: str = field(
        default_factory=lambda: os.getenv("LOG_LEVEL", "INFO")
    )
    dedup_strategy: str = field(
        default_factory=lambda: os.getenv("DEDUP_STRATEGY", "skip")
    )
    prompt_strategy: str = field(
        default_factory=lambda: os.getenv("PROMPT_STRATEGY", "strict")
    )

    def validate(self) -> None:
        """
        校验必填字段是否已正确配置，若缺失则抛出 ValueError 并列出所有缺失项。
        """
        required_fields = {
            "LLM_API_KEY": self.llm_api_key,
            "LLM_MODEL": self.llm_model,
            "LLM_BASE_URL": self.llm_base_url,
        }
        missing = [name for name, value in required_fields.items() if not value]
        if missing:
            raise ValueError(
                f"缺少以下必填配置项: {', '.join(missing)}。"
                f"请在 .env 文件或环境变量中设置它们。"
            )


_config: Optional[Config] = None


def get_config() -> Config:
    """
    获取 Config 单例实例，首次调用时创建并校验配置。
    """
    global _config
    if _config is None:
        _config = Config()
        _config.validate()
    return _config
