
from fastapi import FastAPI
from fastapi.responses import FileResponse
from routers.query import router as query_router
from routers.handler import router as handler_router
from routers.database import router as database_router


app = FastAPI()

@app.get("/", include_in_schema=False)
async def root():
    return FileResponse("templates/index.html")

app.include_router(query_router, prefix="/query", tags=["query"])
app.include_router(handler_router, prefix="/handle", tags=["handle"])
app.include_router(database_router, prefix="/database", tags=["database"])
