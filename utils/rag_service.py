import backoff
from langchain_core.exceptions import OutputParserException
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import PromptTemplate
from langgraph.graph import StateGraph, END
from typing import List, Dict, Any, TypedDict

from . import llm, vector_store
from schemas.query import Answer, ArticleExtraction
from .db_search import search_articles, search_articles_by_numbers

# Define the state for our graph
class GraphState(TypedDict):
    question: str
    top_k: int
    documents: List[Dict[str, Any]]
    article_numbers: List[str]
    db_articles: List[Dict[str, Any]]
    final_answer: Dict[str, Any]

# LLM Parser for Answer
llm_parser = JsonOutputParser(pydantic_object=Answer)

# Prompt Template for Answer
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

# LLM Chain for Answer
llm_chain = prompt | llm | llm_parser

# LLM Parser for Extraction
extraction_parser = JsonOutputParser(pydantic_object=ArticleExtraction)

# Prompt Template for Extraction
extraction_template = """你是一位專業的法律助手。請根據以下參考資料與問題，判斷需要額外查詢哪些法條內容。
請注意參考資料的 metadata 中可能包含 'references' 欄位，其中列出了相關法條編號。
請綜合問題需求與參考資料，列出所有需要查詢的法條編號（僅需數字）。

---
參考資料:
{documents}
---
問題: {question}
---
{format_instructions}
"""

extraction_prompt = PromptTemplate.from_template(
    extraction_template,
    partial_variables={"format_instructions": extraction_parser.get_format_instructions()}
)

# LLM Chain for Extraction
extraction_chain = extraction_prompt | llm | extraction_parser


def retrieve_documents(state: GraphState) -> GraphState:
    """
    Retrieve documents from the vector store.
    """
    print("--- Retrieving Documents ---")
    question = state["question"]
    top_k = state["top_k"]

    # Retrieve documents from two different sources
    law_docs_with_scores = vector_store.similarity_search_with_score(
        question, k=top_k, filter={"source": "labor_law"}
    )
    qa_docs_with_scores = vector_store.similarity_search_with_score(
        question, k=top_k, filter={"source": "labor_law_qa"}
    )

    # Combine and format the documents
    all_docs_with_scores = law_docs_with_scores + qa_docs_with_scores
    
    state["documents"] = [
        {"page_content": doc.page_content, "metadata": doc.metadata, "score": 1 - score}
        for doc, score in all_docs_with_scores
    ]
    print(f"Retrieved {len(state['documents'])} documents.")
    return state

def extract_related_articles(state: GraphState) -> GraphState:
    """
    Extract related article numbers from documents and question.
    """
    print("--- Extracting Related Articles ---")
    question = state["question"]
    documents = state["documents"]
    
    # Prepare context for extraction
    doc_context = "\n\n".join(
        f"Metadata: {doc['metadata']}\nContent: {doc['page_content']}" for doc in documents
    )
    
    try:
        result = extraction_chain.invoke({
            "documents": doc_context,
            "question": question
        })
        article_numbers = result.get("article_numbers", [])
    except Exception as e:
        print(f"Error extracting articles: {e}")
        article_numbers = []
        
    print(f"Extracted article numbers: {article_numbers}")
    state["article_numbers"] = article_numbers
    return state

def search_articles_in_db(state: GraphState) -> GraphState:
    """
    Search for relevant articles in the database.
    """
    print("--- Searching DB Articles ---")
    question = state["question"]
    article_numbers = state.get("article_numbers", [])
    
    # Keyword search
    keyword_articles = search_articles(question)
    
    # Number search
    number_articles = search_articles_by_numbers(article_numbers)
    
    # Combine and deduplicate
    all_articles = keyword_articles + number_articles
    seen_content = set()
    unique_articles = []
    for art in all_articles:
        if art["page_content"] not in seen_content:
            unique_articles.append(art)
            seen_content.add(art["page_content"])
            
    state["db_articles"] = unique_articles
    print(f"Found {len(state['db_articles'])} articles from DB (Keyword: {len(keyword_articles)}, Number: {len(number_articles)}).")
    return state

@backoff.on_exception(backoff.expo, OutputParserException, max_tries=3)
def generate_answer(state: GraphState) -> GraphState:
    """
    Generate the final answer using the retrieved documents and DB articles.
    """
    print("--- Generating Answer ---")
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
    
    print("--- LLM Response ---")
    print(llm_response)

    # Combine all potential references
    all_references = documents + db_articles
    
    # Process hit_references to ensure they contain full content
    raw_hits = llm_response.get("hit_references", [])
    formatted_hit_references = []
    
    for hit in raw_hits:
        matched = False
        # Try to match the hit with one of the retrieved documents
        for ref in all_references:
            # Check if metadata matches. 
            # The LLM might return the metadata object as the hit.
            # We compare the metadata dictionary.
            if ref["metadata"] == hit or ref["metadata"] == hit.get("metadata"):
                formatted_hit_references.append(ref)
                matched = True
                break
        
        if not matched:
            # If no match found, append the hit as is (it might be hallucinated or just metadata)
            formatted_hit_references.append(hit)

    state["final_answer"] = {
        "question": question,
        "answer": llm_response["answer"],
        "hit_references": formatted_hit_references,
        "references": all_references,
    }
    return state

# Define the graph
workflow = StateGraph(GraphState)
workflow.add_node("retrieve_documents", retrieve_documents)
workflow.add_node("extract_related_articles", extract_related_articles)
workflow.add_node("search_articles_in_db", search_articles_in_db)
workflow.add_node("generate_answer", generate_answer)

# Set the entry point and build the graph
workflow.set_entry_point("retrieve_documents")
workflow.add_edge("retrieve_documents", "extract_related_articles")
workflow.add_edge("extract_related_articles", "search_articles_in_db")
workflow.add_edge("search_articles_in_db", "generate_answer")
workflow.add_edge("generate_answer", END)

# Compile the graph
app = workflow.compile() 

def get_rag_result(question: str, top_k: int = 5) -> dict:
    """
    Run the RAG graph to get the result.
    """
    print(f"--- Starting RAG for question: {question} ---")
    inputs = {
        "question": question,
        "top_k": top_k,
    }
    result = app.invoke(inputs)
    final_answer = result.get("final_answer", {})
    print("--- Final RAG Result ---")
    print(final_answer)
    return final_answer
