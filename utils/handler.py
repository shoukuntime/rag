from langchain_community.document_loaders import DirectoryLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters.character import RecursiveCharacterTextSplitter
from langchain_postgres import PGVector

from env_settings import EnvSettings

env_settings = EnvSettings()

embeddings = GoogleGenerativeAIEmbeddings(model=env_settings.EMBEDDING_MODEL, google_api_key=env_settings.GOOGLE_API_KEY)


def ingest_data():
    reader = DirectoryLoader('../data')
    documents = reader.load()
    print(f"Loaded {len(documents)} document(s).")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=env_settings.CHUNK_SIZE,
        chunk_overlap=env_settings.CHUNK_OVERLAP
    )
    docs = text_splitter.split_documents(documents)
    print(f"Split into {len(docs)} chunks.")

    store = PGVector(
        collection_name=env_settings.COLLECTION_NAME,
        connection=env_settings.POSTGRES_URI,
        embeddings=embeddings,
    )

    store.add_documents(docs)
    print("儲存完成!")


if __name__ == "__main__":
    ingest_data()
