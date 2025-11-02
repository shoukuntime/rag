# 勞基法問答 RAG API

這是一個基於檢索增強生成（RAG）技術的問答 API，專門用於回答與勞基法相關的問題。

## 技術棧

*   **FastAPI**: 用於建立 API 的非同步 Web 框架。
*   **LlamaIndex**: 一個用於建立和查詢索引的框架，是本專案 RAG 功能的核心。
*   **MongoDB Atlas**: 作為向量資料庫，儲存文本資料的向量表示。
*   **Hugging Face Embeddings**: 使用 `BAAI/bge-small-zh-v1.5` 模型來產生文本的嵌入向量。
*   **Google Gemini**: 使用 `gemini-1.5-flash` 模型來根據檢索到的資訊產生答案。

## 專案結構

```
.
├── data/                  # 存放勞基法文本資料
├── main.py                # FastAPI 應用程式
├── ingest.py              # 資料導入腳本
├── requirements.txt       # Python 依賴套件
├── env_settings.py        # 環境變數設定
└── README.md              # 專案說明文件
```

## 設定

1.  **安裝依賴**:
    ```bash
    pip install -r requirements.txt
    ```

2.  **設定環境變數**:
    在 `env_settings.py` 檔案中設定您的 `GOOGLE_API_KEY` 和 `MONGO_URI`。

## 使用方法

1.  **資料導入**:
    執行 `ingest.py` 腳本，將 `data/` 目錄下的文本資料轉換為向量並儲存到 MongoDB Atlas。
    ```bash
    python ingest.py
    ```

2.  **啟動 API**:
    使用 `uvicorn` 啟動 FastAPI 應用程式。
    ```bash
    uvicorn main:app --reload
    ```

3.  **發送查詢**:
    使用 `curl` 或其他 API 工具向 `/query` 端點發送 POST 請求。

    ```bash
    curl -X POST "http://127.0.0.1:8000/query" -H "Content-Type: application/json" -d '{"question": "試用期最長多久？"}'
    ```

## API 端點

*   `GET /`: 歡迎訊息。
*   `POST /query`: 接收一個包含 `question` 欄位的 JSON，並回傳一個包含 `response` 欄位的 JSON。
