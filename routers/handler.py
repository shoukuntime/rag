from fastapi import APIRouter, UploadFile, File
import pdfplumber
from utils.handler import ingest_data

router = APIRouter()


@router.post("")
def handle_data_ingestion(file: UploadFile = File(...)):
    with pdfplumber.open(file.file) as pdf:
        text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
    text = text.strip()
    ingest_data(text)
    return {"message": "Data ingestion started successfully."}
