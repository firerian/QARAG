from typing import Any, Optional

from langchain_ollama import OllamaEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
from modules.config import get_config
from modules.logger import get_logger
from modules.prompts import build_prompt

load_dotenv()

logger = get_logger(__name__)

config = get_config()

embeddingllm = OllamaEmbeddings(model=config.embedding_model)
llm = ChatOpenAI(
    api_key=config.llm_api_key,
    model=config.llm_model,
    base_url=config.llm_base_url,
    temperature=config.temperature,
    max_tokens=config.max_tokens
)

llm_retry = retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((Exception,)),
    reraise=True
)


@llm_retry
def _invoke_llm(_llm, prompt: str) -> Any:
    return _llm.invoke(prompt)


def check_embedding_health() -> bool:
    try:
        embeddingllm.embed_query("health_check")
        logger.info("Embedding 服务健康检查通过")
        return True
    except Exception as e:
        logger.error(f"Embedding 服务健康检查失败: {e}")
        return False


def get_llm_answer(vector_db, user_query, llm=llm, retriever_type: str = "hybrid", prompt_strategy: str = "strict", top_k: int = 5) -> Optional[str]:
    if retriever_type == "hybrid":
        results = vector_db.hybrid_search(user_query, top_k)
        if results:
            contents = '\n'.join(results)
            logger.info("--- 检索到的相关片段 ---")
            logger.debug(contents)
            logger.info('-' * 100)
        else:
            contents = ""
            logger.info("未检索到相关文档！")
    elif retriever_type == "default":
        results = vector_db.search(user_query, top_k)
        if results['documents'] and results['documents'][0]:
            contents = '\n'.join(results['documents'][0])
            logger.info("--- 检索到的相关片段 ---")
            logger.debug(contents)
            logger.info('-' * 100)
        else:
            contents = ""
            logger.info("未检索到相关文档！")
    else:
        raise ValueError(f"不支持的 retriever_type: {retriever_type}，可选值为 'hybrid' 或 'default'")

    prompt = build_prompt(strategy=prompt_strategy, contents=contents, user_query=user_query)
    logger.info(prompt)
    try:
        logger.info("--- AI 回答 ---")
        response_text = _invoke_llm(llm, prompt)
        logger.info(response_text.content)
        token_usage = response_text.response_metadata.get("token_usage", {})
        if token_usage:
            prompt_tokens = token_usage.get("prompt_tokens", 0)
            completion_tokens = token_usage.get("completion_tokens", 0)
            total_tokens = token_usage.get("total_tokens", 0)
            logger.info(f"Token 消耗 - prompt: {prompt_tokens}, completion: {completion_tokens}, total: {total_tokens}")
    except Exception as e:
        logger.error(f"调用 LLM 接口失败: {e}")
        logger.warning("所有重试均失败，返回降级消息")
        return "抱歉，AI 服务暂时不可用，请稍后重试。"


def get_llm_answer2(vector_db, user_query, llm=llm, prompt_strategy: str = "strict") -> Optional[str]:
    logger.warning("get_llm_answer2 已弃用，请使用 get_llm_answer(vdb, q, retriever_type='hybrid')")
    return get_llm_answer(vector_db, user_query, llm=llm, retriever_type="hybrid", prompt_strategy=prompt_strategy)
