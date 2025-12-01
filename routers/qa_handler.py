from fastapi import APIRouter, UploadFile, File
import pdfplumber
from utils.qa_handler import ingest_qa_data

router = APIRouter()


@router.post("")
def handle_qa_data_ingestion(file: UploadFile = File(...)):
    """
    處理QA格式的PDF文件，提取文本並進行數據攝取。
    """
    try:
        with pdfplumber.open(file.file) as pdf:
            text = '\n'.join(page.extract_text() or '' for page in pdf.pages)
        text = text.strip()
        
        if not text:
            return {"message": "The provided PDF is empty or could not be read."}

        ingest_qa_data(text)
        return {"message": "QA data ingestion started successfully."}
    except Exception as e:
        return {"message": f"An error occurred: {str(e)}"}
