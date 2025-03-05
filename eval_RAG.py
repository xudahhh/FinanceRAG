import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
import warnings
warnings.filterwarnings('ignore')

# 配置参数
PERSIST_DIR = "./chroma_db"  # 向量数据库存储路径
EMBEDDING_MODEL = "./embedding/BAAI/bge-small-zh-v1___5"  # 需提前下载好
OLLAMA_MODEL = "deepseek-r1:1.5b"  # 需提前用ollama pull下载

def load_vector_db():
    """加载已有向量数据库"""
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},  # GPU可用时可改为cuda
        encode_kwargs={'normalize_embeddings': True}
    )
    return Chroma(
        persist_directory=PERSIST_DIR,
        embedding_function=embeddings
    )

def test_vector_db(vector_db):
    """测试向量数据库内容"""
    # 基本信息
    collection = vector_db._client.get_collection(vector_db._collection.name)
    print(f"\n\033[1m向量数据库基本信息：\033[0m")
    print(f"- 文档总数：{collection.count()}")
    print(f"- 前3条元数据字段示例：{collection.get(limit=3)['metadatas'][0].keys()}")
    
    # 随机抽样文档
    print(f"\n\033[1m随机抽样文档内容：\033[0m")
    sample_docs = collection.get(limit=2)
    for i, (content, meta) in enumerate(zip(sample_docs['documents'], sample_docs['metadatas'])):
        print(f"\n文档 {i+1}:")
        print(f"来源：{meta['source']}")
        print(f"日期：{meta['date']}")
        print(f"内容片段：{content[:150]}...")  # 显示前150个字符

    # 测试检索功能
    print(f"\n\033[1m测试检索功能：\033[0m")
    test_query = "金融分析"
    results = vector_db.similarity_search(test_query, k=2)
    print(f"查询 '{test_query}' 的Top 2结果：")
    for j, doc in enumerate(results):
        print(f"\n结果 {j+1}:")
        print(f"来源：{doc.metadata['source']}")
        print(f"日期：{doc.metadata['date']}")
        print(f"相关度：{doc.metadata.get('relevance_score', 'N/A')}")
        print(f"内容片段：{doc.page_content[:150]}...")

def build_rag_system(vector_db):
    """构建RAG系统"""
    # 初始化LLM
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model=OLLAMA_MODEL,
        temperature=0.7,
        num_ctx=4096
    )
    
    # 构建检索链
    retriever = vector_db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 5, "lambda_mult": 0.25}
    )
    
    # 设计Prompt模板
    prompt_template = """
    作为资深金融分析师，请基于以下背景信息回答问题：
    
    <背景>
    {context}
    </背景>
    
    问题：{question}

    注意：用中文回答，保持专业简洁"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 构建处理链
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

def format_docs(docs):
    return "\n\n".join(
        f"[来源：{doc.metadata['source']} | 日期：{doc.metadata['date']}]\n"
        f"{doc.page_content}" 
        for doc in docs
    )

if __name__ == "__main__":
    # 加载已有数据库
    print("正在加载向量数据库...")
    vector_db = load_vector_db()
    
    # 测试数据库内容
    # test_vector_db(vector_db)
    
    # 构建RAG系统
    print("\n\033[1m构建RAG系统...\033[0m")
    rag_chain = build_rag_system(vector_db)
    
    # 测试问答
    print("\n\033[1m测试问答功能：\033[0m")
    queries = [
        "什么是deepseek？为何deepseek这么火爆？股票可以买哪些股？",
        "近期有哪些值得关注的金融趋势？",
        "请分析当前宏观经济形势对股市的影响"
    ]
    
    for query in queries:
        print(f"\n\033[1m问题：{query}\033[0m")
        response = rag_chain.invoke(query)
        print(f"\033[1;34m回答：\n{response}\033[0m")
        print("-"*50)