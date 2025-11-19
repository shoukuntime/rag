from fastapi import APIRouter
from utils.handler import ingest_data

router = APIRouter()


@router.post("/")
def handle_data_ingestion():
    ingest_data()
    return {"message": "Data ingestion started successfully."}
