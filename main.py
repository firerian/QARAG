from modules import DataProcessor,MyVectorDBConnector,get_llm_answer,embeddingllm,get_llm_answer2








if __name__ == '__main__':
    vector_db = MyVectorDBConnector(collection_name="demo", client=embeddingllm)




    # processor = DataProcessor(chunk_size=200, chunk_overlap=40)
    # # 1. 读取问答对并存入向量库
    # try:
    #     instructions, outputs = processor.load_qa_json('./Data/train.json')
    #     vector_db.add_documents(instructions, outputs)  # 假设 rag 是你之前封装好的 LocalRAGSystem 实例
    # except Exception as e:
    #     print(e)
    # # 2. 读取长文本并存入向量库
    # try:
    #     text_chunks = processor.load_and_split_text('./Data/deepseek百度百科.txt')
    #     # 这里你可以在 LocalRAGSystem 里加一个直接存入文本列表的方法，或者遍历存入
    #     vector_db.add_documents(text_chunks)
    # except Exception as e:
    #     print(e)
    # print("数据库内的数据为：" + str(vector_db.get(count=1)))
    print("普通检索")
    get_llm_answer(vector_db,"deepseek相继发布了哪些模型？")
    print('0' * 40)
    print('0' * 40)
    print("混合检索")
    get_llm_answer2(vector_db, "deepseek相继发布了哪些模型？")
    # get_llm_answer2(vector_db, "deepseek相继发布了哪些模型？")










