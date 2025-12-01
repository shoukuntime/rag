from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from . import llm, vector_store
from schemas.query import Answer


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


llm_chain = prompt | llm | llm_parser


def get_rag_result(question: str, top_k: int = 5) -> dict:
    docs_with_scores = vector_store.similarity_search_with_score(question, k=top_k)

    docs = [doc for doc, score in docs_with_scores]
    context = "\n\n".join(
        f"Source:\n{doc.metadata}\n\nContent:\n{doc.page_content}" for doc in docs
    )
    references = [{"page_content": doc.page_content, "metadata": doc.metadata, "score": score} for doc, score in docs_with_scores]

    llm_response = llm_chain.invoke({
        "context": context,
        "question": question
    })

    return {
        "question": question,
        "answer": llm_response["answer"],
        "reference": references
    }
