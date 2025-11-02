from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional

from services.rag_service import get_rag_result

router = APIRouter()

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(5, ge=1, le=10, description="要檢索的文件數量。")

@router.post("/query")
def query_labor_law(request: QueryRequest):
    result = get_rag_result(request.question, top_k=request.top_k)
    return {"response": result}

@router.get("/")
def read_root():
    return {"message": "歡迎使用《勞基法》RAG API"}
