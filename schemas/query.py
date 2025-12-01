from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any

class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = Field(5, ge=1, le=10, description="要檢索的文件數量。")

class Answer(BaseModel):
    answer: str = Field(description="問題的答案。")
    hit_references: List[Dict[str, Any]] = Field(description="生成答案所引用的相關參考資料。")
