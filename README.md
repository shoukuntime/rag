# 勞動法規 RAG 系統 (Labor Law RAG System)

本專案是一個檢索增強生成 (RAG) 系統，專為回答勞動法規相關問題而設計。後端採用 FastAPI，使用 PostgreSQL 搭配 `pgvector` 進行向量儲存，並利用 Google 的生成式 AI 模型 (Gemini) 進行文本嵌入與生成。

## 功能特色

- **RAG 查詢 (RAG Query)**：針對勞動法規進行提問，系統會從向量資料庫中檢索相關上下文來回答問題。
- **直接提問 (Direct Ask)**：直接向 LLM 提問一般性問題，不經過檢索流程。
- **資料攝取 (Data Ingestion)**：支援上傳 PDF 檔案（勞動法規文本或 QA 問答格式），將資料處理並匯入向量資料庫。
- **資料庫管理 (Database Management)**：可查看目前儲存的向量資料，或清除資料庫內容。
- **容器化部署 (Dockerized)**：支援使用 Docker Compose 快速部署應用程式與資料庫。

## 技術棧

- **後端框架**: FastAPI
- **LLM 框架**: LangChain
- **向量資料庫**: PostgreSQL (pgvector)
- **AI 模型**: Google Generative AI (Gemini)
- **容器化**: Docker & Docker Compose

## 前置需求

- Docker & Docker Compose
- Python 3.11 (若需本地開發)
- Google API Key

## 設定與安裝

### 1. 複製專案 (Clone the Repository)

```bash
git clone <repository-url>
cd rag
```

### 2. 環境變數設定 (Environment Configuration)

在專案根目錄下建立一個 `.env` 檔案，並填入以下變數：

```env
GOOGLE_API_KEY=
MODEL_NAME=
EMBEDDING_MODEL=

POSTGRES_URI=
POSTGRES_USER=
POSTGRES_PASSWORD=
POSTGRES_DB=
COLLECTION_NAME=

TOP_K=
```

### 3. 使用 Docker Compose 啟動

啟動應用程式與資料庫：

```bash
docker-compose up -d --build
```

- API 服務將在 `http://localhost:8000` 上運行。
- Adminer (資料庫管理介面) 將在 `http://localhost:8080` 上運行。

### 4. 本地開發 (可選)

如果您希望在本地執行 Python 應用程式，但資料庫仍使用 Docker：

1.  啟動資料庫：
    ```bash
    docker-compose up -d db
    ```
2.  安裝依賴套件：
    ```bash
    pip install -r requirements.txt
    ```
3.  執行應用程式：
    ```bash
    uvicorn main:app --reload
    ```
    *注意：本地執行時，請將 `.env` 中的 `POSTGRES_URI` 主機名稱從 `db` 改為 `localhost`。*

## API 端點說明

### 查詢 (Query)
- `POST /query/rag`: 使用 RAG 模式詢問問題。
- `POST /query/ask`: 直接向 LLM 詢問問題。

### 資料處理 (Data Handler)
- `POST /handle/labor_law`: 上傳 PDF 檔案以攝取勞動法規文本。
- `POST /handle/labor_law_qa`: 上傳包含 QA 問答對的 PDF 檔案。

### 資料庫 (Database)
- `GET /database`: 查看所有儲存的嵌入向量資料。
- `DELETE /database/clear`: 清除資料庫中的所有嵌入向量。

## 專案結構

```
.
├── data/               # 資料儲存目錄
├── routers/            # API 路由 (query, handler, database)
├── schemas/            # Pydantic 模型定義
├── templates/          # HTML 模板
├── utils/              # 輔助功能 (RAG 服務, 資料攝取邏輯)
├── docker-compose.yml  # Docker 服務配置
├── Dockerfile          # API 容器定義
├── env_settings.py     # 環境變數管理
├── main.py             # 應用程式進入點
└── requirements.txt    # Python 依賴列表
```
