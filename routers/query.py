from fastapi import APIRouter
from utils.rag_service import get_rag_result
from schemas.query import QueryRequest

router = APIRouter()

@router.post("/")
def query_labor_law(request: QueryRequest):
    result = get_rag_result(request.question, top_k=request.top_k)
    return {"response": result}
