# 使用官方 Python 映像檔
FROM python:3.9-slim

# 設定工作目錄
WORKDIR /app

# 複製相依套件檔案
COPY requirements.txt requirements.txt

# 安裝相依套件
RUN pip install --no-cache-dir -r requirements.txt

# 複製專案的其餘部分
COPY . .

# 執行應用程式
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
