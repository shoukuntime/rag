from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from . import llm, vector_store
from schemas.query import Answer

llm_parser = JsonOutputParser(pydantic_object=Answer)

template = """你是一位專業大法官。請僅根據以下<<勞動基準法>>的內容回答問題，若無法回答請如實回覆。
---
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


def get_ask_result(question: str) -> dict:
    with open('data/labor_law.txt', 'r', encoding='utf-8') as f:
        context = f.read()
    llm_response = llm_chain.invoke({
        "context": context,
        "question": question
    })
    return {
        "question": question,
        "answer": llm_response["answer"],
    }
