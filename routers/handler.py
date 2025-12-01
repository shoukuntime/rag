from fastapi import APIRouter, UploadFile, File
from typing import List
import pdfplumber
from utils.handler import ingest_data

router = APIRouter()


@router.post("")
def handle_data_ingestion(files: List[UploadFile] = File(...)):
    for file in files:
        with pdfplumber.open(file.file) as pdf:
            text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        text = text.strip()
        if text:
            ingest_data(text)
    return {"message": "Data ingestion started successfully for all files."}
