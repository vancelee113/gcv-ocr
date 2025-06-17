# Streamlit GCV App

這是一個使用 Streamlit 執行的 Google Cloud Vision API 範例。

## 🚀 如何執行

```bash
# 安裝虛擬環境
python -m venv venv
source venv/bin/activate  # Mac/Linux
# 或 venv\Scripts\activate  # Windows

# 安裝依賴
pip install -r requirements.txt

# 執行 Streamlit
streamlit run app.py
```

## 🔑 設定金鑰

請將 Google Cloud Vision 的金鑰 (JSON) 放在專案目錄，並在 .env 設定路徑：

```plaintext
GOOGLE_APPLICATION_CREDENTIALS=./my_gcv_key.json
```

## 📂 結構

- `app.py`：主要 Streamlit 應用程式
- `requirements.txt`：所需 Python 套件
- `.gitignore`：Git 忽略設定
- `.env.example`：範例環境變數"# force deploy" 
