
from fastapi import FastAPI
from routers.query import router as query_router
from routers.handler import router as handler_router


app = FastAPI()

app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(handler_router, prefix="/handle", tags=["handle"])
