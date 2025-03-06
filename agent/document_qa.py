import os

from chromadb.api.types import IncludeEnum
from langchain.chains import RetrievalQA
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM

# 🛠 配置部分
persist_directory = "./chroma_db"  # Chroma 本地持久化目录
embeddings = HuggingFaceEmbeddings(
    model_name=r'C:\Users\huangrx\.cache\huggingface\hub\bge-large-zh-v1.5',
    model_kwargs={'device': 'cuda'},
    encode_kwargs={'normalize_embeddings': True}
)

# 🟢 初始化本地向量数据库（如果已存在则加载）
if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

vector_store = Chroma(
    persist_directory=persist_directory,
    embedding_function=embeddings
)

# 🌐 LLM 初始化
llm = OllamaLLM(
    base_url='http://localhost:11434',
    model='deepseek-r1:14b'
)


# 📁 计算上传顺序的函数 (自动计算 upload_order)
def get_next_upload_order(user_id, session_id):
    # 获取所有文档
    results = vector_store._collection.get(
        limit=100,  # 限制检索数量
        include=[IncludeEnum.metadatas]
    )

    # 获取符合条件的文档，并计算最大 upload_order
    current_orders = []
    for metadata in results["metadatas"]:
        if metadata.get("user_id") == user_id and metadata.get("session_id") == session_id:
            current_orders.append(metadata.get("upload_order", 0))

    # 计算下一个上传顺序
    next_order = max(current_orders, default=0) + 1
    return next_order


# 📁 上传文档的函数 (自动获取 upload_order)
def upload_file(file_path, user_id, session_id):
    # 自动获取当前用户和会话的下一个上传顺序
    upload_order = get_next_upload_order(user_id, session_id)

    # 加载和分割文档
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_docs = text_splitter.split_documents(documents)

    # 添加元数据 (自动计算的 upload_order)
    for doc in split_docs:
        doc.metadata["user_id"] = user_id
        doc.metadata["session_id"] = session_id
        doc.metadata["file_name"] = os.path.basename(file_path)
        doc.metadata["upload_order"] = upload_order

    # 添加文档到向量数据库并持久化
    vector_store.add_documents(split_docs)
    print(f"📥 文件 '{file_path}' 已上传并存储到数据库！ (upload_order={upload_order})")


# 🔍 用户查询的函数
def query_user_data(user_id, session_id=None, file_name=None, upload_order=None, query_text=""):
    # 构建检索过滤条件
    filter_conditions = [{"user_id": user_id}]

    if session_id:
        filter_conditions.append({"session_id": session_id})
    if file_name:
        filter_conditions.append({"file_name": file_name})
    if upload_order:
        filter_conditions.append({"upload_order": {"$eq": upload_order}})

    # 使用 $and 组合所有条件
    filter_condition = {"$and": filter_conditions} if len(filter_conditions) > 1 else filter_conditions[0]

    # 仅检索符合条件的文档
    retriever = vector_store.as_retriever(search_kwargs={"filter": filter_condition})
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    # 执行查询
    result = qa_chain.invoke({"query": query_text})

    print("\n💬 问题:", query_text)
    print("📖 回答:", result['result'])

    # 展示相关文档信息
    for doc in result['source_documents']:
        print(f"\n📎 来源文件: {doc.metadata.get('file_name', '未知')}")
        print(f"🆔 用户ID: {doc.metadata.get('user_id', '未知')}")
        print(f"📄 上传顺序: {doc.metadata.get('upload_order', '未知')}")
        print(f"📑 文档片段: {doc.page_content[:200]}...")


# 🚀 示例: 上传多个用户的文件 (自动分配 upload_order)
# upload_file("人物传记-慕勒小传.pdf", user_id="user_123", session_id="session_123")
# upload_file("古道书房-巴刻文集.pdf", user_id="user_123", session_id="session_123")
# upload_file("人生学校.pdf", user_id="user_456", session_id="session_125")

# 🔎 示例: 查询特定用户的文档
query_user_data(
    user_id="user_123",
    session_id="session_123",
    query_text="第一次上传的文件中主要内容是什么？",
    upload_order=1
)

# 🔎 示例: 查询特定上传顺序的文档
query_user_data(
    user_id="user_123",
    session_id="session_123",
    upload_order=2,
    query_text="第二次上传的文件里有哪些内容？"
)
