import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import warnings
warnings.filterwarnings('ignore')

# 配置参数
DATA_PATH = "公众号数据v2.xlsx"  # 假设包含title,content,date三列
PERSIST_DIR = "./chroma_db"  # 向量数据库存储路径
EMBEDDING_MODEL = "./embedding/BAAI/bge-small-zh-v1___5"  # 需提前下载好
OLLAMA_MODEL = "deepseek-r1:1.5b"  # 需提前用ollama pull下载

# 数据预处理管道
def data_pipeline():
    # 读取数据
    df = pd.read_excel(DATA_PATH)
    
    # 数据清洗
    df = df[df['content'].str.len().between(500, 10000)]  # 保留合理长度内容
    df = df.dropna(subset=['content'])
    
    # 时间过滤（保留最近2年）
    df['date'] = pd.to_datetime(df['date'])
    cutoff_date = pd.Timestamp.now() - pd.DateOffset(years=2)
    df = df[df['date'] > cutoff_date]
    
    # 文本分块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", "。", "！", "？"]
    )
    
    documents = []
    for _, row in df.iterrows():
        chunks = text_splitter.split_text(row['content'])
        for chunk in chunks:
            documents.append(Document(  # 直接创建Document对象
                page_content=chunk,
                metadata={
                    "source": row['title'],
                    "date": row['date'].strftime("%Y-%m-%d")
                }
            ))
    
    return documents

# 构建RAG系统
def build_rag_system(documents):
    # 1. 初始化嵌入模型
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cuda'},  # GPU可用时可改为cuda
        encode_kwargs={'normalize_embeddings': True}
    )
    
    # 2. 创建向量数据库
    # new_documents = [doc["page_content"] for doc in documents]
    # new_metadatas = [doc["metadata"] for doc in documents]

    vector_db = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=PERSIST_DIR
    )
    vector_db.persist()
    
    # 3. 初始化LLM
    llm = ChatOllama(
        base_url="http://localhost:11434",
        model=OLLAMA_MODEL,
        temperature=0.7,
        num_ctx=4096
    )
    
    # 4. 构建检索链
    retriever = vector_db.as_retriever(
        search_type="mmr",  # 最大边际相关性搜索
        search_kwargs={"k": 5, "lambda_mult": 0.25}
    )
    
    # 5. 设计Prompt模板
    prompt_template = """
    作为资深金融分析师，请基于以下背景信息回答问题：
    
    <背景>
    {context}
    </背景>
    
    问题：{question}

    注意：用中文回答，保持专业简洁"""
    
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # 6. 构建处理链
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

# 格式化检索结果
def format_docs(docs):
    return "\n\n".join(
        f"[来源：{doc.metadata['source']} | 日期：{doc.metadata['date']}]\n"
        f"{doc.page_content}" 
        for doc in docs
    )

# 运行测试
if __name__ == "__main__":
    # 初始化RAG系统
    print("正在初始化RAG系统...")
    docs = data_pipeline()
    rag_chain = build_rag_system(docs)
    query = ("什么是deepseek？为何deepseek这么火爆？股票可以买哪些股？")
    print(f"\n\033[1m问题：{query}\033[0m")
    response = rag_chain.invoke(query)
    print(f"\033[1;34m回答：\n{response}\033[0m")
    print("-"*50)

    # 示例查询
    # while True:
    #     query = input("问题：")
    #     # print(f"\n\033[1m问题：{query}\033[0m")
    #     response = rag_chain.invoke(query)
    #     print(f"\033[1;34m回答：\n{response}\033[0m")
    #     print("-"*50)
