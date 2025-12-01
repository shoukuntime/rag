import backoff
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict

from . import llm, vector_store
from schemas.query import Answer
from .db_search import search_articles

# Define the state for our graph
class GraphState(TypedDict):
    question: str
    top_k: int
    documents: List[Dict[str, Any]]
    db_articles: List[Dict[str, Any]]
    final_answer: Dict[str, Any]

# LLM Parser
llm_parser = JsonOutputParser(pydantic_object=Answer)

# Prompt Template
template = """你是一位回答問題的助手。請根據以下參考資料與法條內容回答問題。
---
參考資料:
{documents}
---
法條內容:
{db_articles}
---
問題: {question}
---
{format_instructions}
"""
prompt = PromptTemplate.from_template(
    template,
    partial_variables={"format_instructions": llm_parser.get_format_instructions()}
)

# LLM Chain
llm_chain = prompt | llm | llm_parser

def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieve documents from the vector store.
    """
    question = state["question"]
    top_k = state["top_k"]
    

    docs_with_scores = vector_store.similarity_search_with_score(question, k=top_k)
    
    state["documents"] = [
        {"page_content": doc.page_content, "metadata": doc.metadata, "score": score}
        for doc, score in docs_with_scores
    ]
    return state

def search_articles_in_db(state: GraphState) -> GraphState:
    """
    Search for relevant articles in the database.
    """
    question = state["question"]
    state["db_articles"] = search_articles(question)
    return state

@backoff.on_exception(backoff.expo, OutputParserException, max_tries=3)
def generate_answer(state: GraphState) -> GraphState:
    """
    Generate the final answer using the retrieved documents and DB articles.
    """
    question = state["question"]
    documents = state["documents"]
    db_articles = state["db_articles"]

    # Format the context for the LLM
    doc_context = "\n\n".join(
        f"Source: {doc['metadata']}\nContent: {doc['page_content']}" for doc in documents
    )
    article_context = "\n\n".join(
        f"Source: {art['metadata']}\nContent: {art['page_content']}" for art in db_articles
    )

    llm_response = llm_chain.invoke({
        "documents": doc_context,
        "db_articles": article_context,
        "question": question,
    })

    state["final_answer"] = {
        "question": question,
        "answer": llm_response["answer"],
        "hit_references": llm_response.get("hit_references", []),
        "reference": documents,  # Keep original references from vector store
    }
    return state

# Define the graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("search_articles_in_db", search_articles_in_db)
workflow.add_node("generate_answer", generate_answer)

# Set the entry point and build the graph
workflow.set_entry_point("retrieve_documents")
workflow.add_edge("retrieve_documents", "search_articles_in_db")
workflow.add_edge("search_articles_in_db", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile the graph
app = workflow.compile()

def get_rag_result(question: str, top_k: int = 5) -> dict:
    """
    Run the RAG graph to get the result.
    """
    inputs = {
        "question": question,
        "top_k": top_k,
    }
    result = app.invoke(inputs)
    return result.get("final_answer", {})
