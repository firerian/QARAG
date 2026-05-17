import os
from langchain_ollama import ChatOllama,OllamaEmbeddings
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
load_dotenv()



EMBEDDING_MODEL = "bge-m3:567m"  # 嵌入模型
CHAT_MODEL = "deepseek-r1:1.5b"  # 聊天模型（请确保你本地有这个模型，可以用 ollama list 查看）


embeddingllm = OllamaEmbeddings(model=EMBEDDING_MODEL)
llm =  ChatOpenAI(
    api_key = os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    base_url=os.getenv("LLM_BASE_URL"),
    temperature = os.getenv("TEMPERATURE"),
    max_tokens = os.getenv("MAX_TOKENS")
)


def get_llm_answer(vector_db,user_query,llm = llm):
    results = vector_db.search(user_query, 5)
    # 提取相关文档内容
    if results['documents'] and results['documents'][0]:
        contents = '\n'.join(results['documents'][0])
        print("\n--- 检索到的相关片段 ---")
        print(contents)
        print('-' * 100)
    else:
        contents = ""
        print("未检索到相关文档！")
    # 5. 构造 Prompt
    prompt = f"""
   你是一个严格的问答机器人。请严格遵守以下规则：
    1. 严禁使用你自身的训练数据、常识或外部知识进行回答,不要自行总结，严格按照【已知信息】来回答问题。
    2. 如果【已知信息】中没有直接包含回答用户问题所需的内容，你必须且只能回复：“我无法回答您的问题”。
    3. 即使你知道答案，但只要【已知信息】里没有，就视为不知道。
    【已知信息】
    {contents}
    ----
    用户问：
    {user_query}
    请用中文回答用户问题。
    """
    print(prompt)
    try:
        print("\n--- AI 回答 ---")
        response_text = llm.invoke(prompt)
        print(response_text.content)
    except Exception as e:
        print(f"调用 Ollama 聊天接口失败: {e}")
        print("请确保 Ollama 正在运行，且已拉取模型: ollama pull " + CHAT_MODEL)




def get_llm_answer2(vector_db,user_query,llm = llm):
    results = vector_db.hybrid_search(user_query, 5)
    # 提取相关文档内容
    if results:
        contents = '\n'.join(results)
        print("\n--- 检索到的相关片段 ---")
        print(contents)
        print('-' * 100)
    else:
        contents = ""
        print("未检索到相关文档！")
    # 5. 构造 Prompt
    prompt = f"""
    你是一个严格的问答机器人。请严格遵守以下规则：
    1. 严禁使用你自身的训练数据、常识或外部知识进行回答,不要自行总结，严格按照【已知信息】来回答问题。
    2. 如果【已知信息】中没有直接包含回答用户问题所需的内容，你必须且只能回复：“我无法回答您的问题”。
    3. 即使你知道答案，但只要【已知信息】里没有，就视为不知道。
    【已知信息】
    {contents}
    ----
    用户问：
    {user_query}
    请用中文回答用户问题。
    """
    print(prompt)
    try:
        print("\n--- AI 回答 ---")
        response_text = llm.invoke(prompt)
        print(response_text.content)
    except Exception as e:
        print(f"调用 Ollama 聊天接口失败: {e}")



