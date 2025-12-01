from fastapi import APIRouter, UploadFile, File
import pdfplumber
from utils.qa_handler import ingest_qa_data
from utils import vector_store

router = APIRouter()


@router.get("")
def get_all_qa():
    """
    獲取所有 source 為 'labor_law_qa' 的問答對。
    """
    # 使用 similarity_search 配合 metadata 過濾來獲取所有相關文件
    # 我們傳入一個空查詢，並將 k 設為一個較大的數值以獲取所有可能的項目
    try:
        results = vector_store.similarity_search(
            query=" ",
            k=1000,  # 假設 QA 項目總數小於 1000
            filter={"source": "labor_law_qa"}
        )
        
        # 將 Document 對象轉換為可序列化的字典
        qa_list = [
            {
                "question": doc.page_content,
                "answer": doc.metadata.get("answer", ""),
                "references": doc.metadata.get("references", [])
            }
            for doc in results
        ]
        return qa_list
    except Exception as e:
        # 這裡可以加入更詳細的日誌記錄
        return {"error": f"Failed to retrieve QA data: {str(e)}"}


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
