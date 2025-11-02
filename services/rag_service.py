import os
import pymongo
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from ..env_settings import EnvSettings

env_settings = EnvSettings()
if not env_settings.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")

# 設置 Gemini LLM
llm = ChatGoogleGenerativeAI(model=env_settings.MODEL_NAME)

# 連接 MongoDB
client = pymongo.MongoClient(env_settings.MONGO_URI)
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# 初始化 vector store
vector_store = MongoDBAtlasVectorSearch(
    collection=client[env_settings.DB_NAME][env_settings.COLLECTION_NAME],
    embedding=GoogleGenerativeAIEmbeddings(model=env_settings.EMBEDDING_MODEL),
    index_name=ATLAS_VECTOR_SEARCH_INDEX_NAME
)

retriever = vector_store.as_retriever()

template = """Answer the question based only on the following context:
{context}

Question: {question}
"""
prompt = ChatPromptTemplate.from_template(template)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)
