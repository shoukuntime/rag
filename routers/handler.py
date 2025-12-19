from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import List
import pdfplumber
from utils.handler import ingest_data, ingest_qa_data

router = APIRouter()


@router.post("/labor_law")
def handle_data_ingestion(file: UploadFile = File(...)):
    with pdfplumber.open(file.file) as pdf:
        text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
    text = text.strip()
    if text:
        ingest_data(text)
    return {"message": "Data ingestion started successfully."}


@router.post("/labor_law_qa")
def handle_qa_data_ingestion(files: List[UploadFile] = File(...)):
    """
    處理QA格式的PDF文件，提取文本並進行數據攝取。
    """
    try:
        for file in files:
            with pdfplumber.open(file.file) as pdf:
                text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
            text = text.strip()

            if not text:
                continue

            ingest_qa_data(text)
        return {"message": "QA data ingestion started successfully for all files."}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
