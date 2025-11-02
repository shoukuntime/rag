
from fastapi import APIRouter
from pydantic import BaseModel

from ..services.rag_service import chain

router = APIRouter()

class QueryRequest(BaseModel):
    question: str

@router.post("/query")
def query_labor_law(request: QueryRequest):
    result = chain.invoke(request.question)
    return {"response": result}

@router.get("/")
def read_root():
    return {"message": "Welcome to the Labor Law RAG API"}
