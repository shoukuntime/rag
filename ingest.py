import os
import pymongo
from langchain_community.document_loaders import DirectoryLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from env_settings import EnvSettings

env_settings = EnvSettings()
if not env_settings.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
os.environ["GOOGLE_API_KEY"] = env_settings.GOOGLE_API_KEY

# 設置嵌入模型
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")


def ingest_data():
    """
    讀取文本資料，建立向量索引，並儲存至 MongoDB。
    """
    if not env_settings.MONGO_URI:
        raise ValueError("MONGO_URI not found in environment variables.")

    # 從 data 資料夾載入文件
    reader = DirectoryLoader('./data')
    documents = reader.load()
    print(f"Loaded {len(documents)} document(s).")

    # 分割文件
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=env_settings.CHUNK_SIZE,
        chunk_overlap=env_settings.CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    # 初始化 MongoDB
    client = pymongo.MongoClient(env_settings.MONGO_URI)
    ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

    # 建立索引，這會自動將向量儲存到 MongoDB
    print("Indexing documents and storing in MongoDB...")
    MongoDBAtlasVectorSearch.from_documents(
        documents=docs,
        embedding=GoogleGenerativeAIEmbeddings(model=env_settings.EMBEDDING_MODEL),
        collection=client[env_settings.DB_NAME][env_settings.COLLECTION_NAME],
        index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME,
    )

    print("Ingestion complete.")


if __name__ == "__main__":
    ingest_data()
