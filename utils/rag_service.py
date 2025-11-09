from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_postgres import PGVector
from pydantic import SecretStr, BaseModel, Field

from env_settings import EnvSettings

env_settings = EnvSettings()

llm = ChatGoogleGenerativeAI(
    model=env_settings.MODEL_NAME,
    google_api_key=SecretStr(env_settings.GOOGLE_API_KEY)
)

embeddings = GoogleGenerativeAIEmbeddings(
    model=env_settings.EMBEDDING_MODEL,
    google_api_key=SecretStr(env_settings.GOOGLE_API_KEY)
)

vector_store = PGVector(
    collection_name=env_settings.COLLECTION_NAME,
    connection=env_settings.POSTGRES_URI,
    embeddings=embeddings,
)


class Answer(BaseModel):
    answer: str = Field(description="問題的答案。")


llm_parser = JsonOutputParser(pydantic_object=Answer)

template = """你是一位回答問題的助手。請僅根據以下參考資料回答問題，若無法回答請如實回覆。
---
參考資料:
{context}
---
問題: {question}
---
{format_instructions}
"""
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": llm_parser.get_format_instructions()}
)


def format_docs_for_prompt(docs):
    return "\n\n".join(doc.page_content for doc in docs)


def get_references(docs_with_scores):
    return [{"page_content": doc.page_content, "score": score} for doc, score in docs_with_scores]


llm_chain = prompt | llm | llm_parser


def get_rag_result(question: str, top_k: int = 5) -> dict:
    docs_with_scores = vector_store.similarity_search_with_score(question, k=top_k)

    docs = [doc for doc, score in docs_with_scores]
    context = format_docs_for_prompt(docs)
    references = get_references(docs_with_scores)

    llm_response = llm_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "question": question,
        "answer": llm_response["answer"],
        "reference": references
    }
