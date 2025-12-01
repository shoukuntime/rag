from fastapi import APIRouter
from schemas.query import QueryRequest
from utils.rag_service import get_rag_result
from utils.ask import get_ask_result


router = APIRouter()

@router.post("")
def query_labor_law(request: QueryRequest):
    result = get_rag_result(request.question, top_k=request.top_k)
    return {"response": result}

@router.post("/ask")
def ask_question(question: str):
    result = get_ask_result(question)
    return {"response": result}
