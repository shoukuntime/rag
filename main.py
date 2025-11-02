
from fastapi import FastAPI
from api.endpoints import router

app = FastAPI()

app.include_router(router)
