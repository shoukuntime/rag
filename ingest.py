import time
import pymongo
from langchain_community.document_loaders import DirectoryLoader
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter

from env_settings import EnvSettings

env_settings = EnvSettings()

embeddings = GoogleGenerativeAIEmbeddings(model=env_settings.EMBEDDING_MODEL)


def ingest_data():
    reader = DirectoryLoader('./data')
    documents = reader.load()
    print(f"Loaded {len(documents)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=env_settings.CHUNK_SIZE,
        chunk_overlap=env_settings.CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    client = pymongo.MongoClient(env_settings.MONGO_URI)
    collection = client[env_settings.DB_NAME][env_settings.COLLECTION_NAME]

    print("建立索引文件並儲存於MongoDB...")
    batch_size = 5
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i+batch_size]
        MongoDBAtlasVectorSearch.from_documents(
            documents=batch,
            embedding=GoogleGenerativeAIEmbeddings(model=env_settings.EMBEDDING_MODEL),
            collection=collection,
            index_name=env_settings.INDEX_NAME,
        )
        print(f"Processed batch {i//batch_size + 1}")
        time.sleep(60)

    print("儲存完成!")


if __name__ == "__main__":
    ingest_data()
