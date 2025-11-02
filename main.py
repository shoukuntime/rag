
import os
from fastapi import FastAPI
from pydantic import BaseModel
import pymongo
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_mongodb import MongoDBAtlasVectorSearch
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough


from env_settings import EnvSettings

# 初始化環境變數
env_settings = EnvSettings()

if not env_settings.GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in environment variables.")
os.environ["GOOGLE_API_KEY"] = env_settings.GOOGLE_API_KEY

# 設置 Gemini LLM
llm = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash")

# 設置嵌入模型
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# 初始化 FastAPI 應用
app = FastAPI()

# 查詢請求的資料模型
class QueryRequest(BaseModel):
    question: str

# 連接 MongoDB
if not env_settings.MONGO_URI:
    raise ValueError("MONGO_URI not found in environment variables.")

client = pymongo.MongoClient(env_settings.MONGO_URI)
DB_NAME = "langchain_db"
COLLECTION_NAME = "test"
MONGODB_COLLECTION = client[DB_NAME][COLLECTION_NAME]
ATLAS_VECTOR_SEARCH_INDEX_NAME = "vector_index"

# 初始化 vector store
vector_store = MongoDBAtlasVectorSearch(
    collection=MONGODB_COLLECTION,
    embedding=embeddings,
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


@app.post("/query")
def query_labor_law(request: QueryRequest):
    """
    接收問題，並從勞基法知識庫中查詢答案。
    """
    result = chain.invoke(request.question)
    return {"response": result}

@app.get("/")
def read_root():
    return {"message": "Welcome to the Labor Law RAG API"}
