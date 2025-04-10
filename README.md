# 🧠 RAG with LangChain, Qdrant, and Gemini

Dự án này là một hệ thống **Retrieval-Augmented Generation (RAG)** hỗ trợ hỏi đáp dựa trên văn bản pháp luật tiếng Việt, tích hợp các công nghệ:
- 🧱 [LangChain](https://www.langchain.com/) để xây dựng pipeline truy xuất thông tin
- 🔍 [Qdrant](https://qdrant.tech/) làm vector database
- ✨ Google Gemini làm LLM backend để sinh câu trả lời

---

## 📦 Cấu trúc thư mục

```plaintext
rag-with-langchain-qdrant-gemini/
├── .streamlit/               # Cấu hình giao diện Streamlit (nếu có)
├── documents/                # Chứa các file văn bản pháp luật đầu vào
├── models/                   # Các mô hình NLP tiếng Việt
│   ├── phobert-base-v2/      # Mô hình PhoBERT
│   └── vncorenlp/            # VnCoreNLP (gồm JAR và model files)
├── notebooks/
│   └── rag.ipynb             # Notebook minh họa pipeline RAG
├── pages/
│   └── file_upload.py        # Giao diện upload tài liệu (cho Streamlit)
├── src/
│   ├── logs/                 # Ghi log quá trình xử lý
│   │   ├── documents.txt
│   │   ├── extracted_docs.txt
│   │   └── metadata.json
│   ├── processor/            # Xử lý văn bản & lưu trữ vector
│   │   ├── document_loader.py
│   │   ├── keyword_extractor.py
│   │   └── vector_store.py
│   ├── rag/                  # Pipeline RAG chính
│   │   ├── generator.py
│   │   ├── rag_pipeline.py
│   │   └── retriever.py
│   └── test/                 # Unit tests
│       ├── test.py
│       ├── test_llm.py
│       └── test_rag_pipeline.py
├── .env                      # Biến môi trường (API key, config)
├── .gitignore
├── docker-compose.yml        # Dùng để triển khai Qdrant hoặc dịch vụ liên quan
├── main.py                   # Điểm khởi động ứng dụng
├── requirements.txt          # Thư viện Python cần thiết
└── README.md                 # File bạn đang đọc
```


## Cài đặt và chạy thử
### Clone project
```plaintext
git clone https://github.com/your-username/rag-with-langchain-qdrant-gemini.git
cd rag-with-langchain-qdrant-gemini
```

### Cài đặt dependencies
```plaintext
python -m venv venv
source venv/bin/activate  # Hoặc `venv\Scripts\activate` nếu dùng Windows
pip install -r requirements.txt
```

### Cấu hình .env
Tạo .env với các biến môi trường như:
```plaintext
GEMINI_API_KEY=your-gemini-api-key
OPENAI_API_KEY=your-openai-api-key
MODEL_NAME=gemini-2.0-flash-exp
EMBEDDING_MODEL_NAME="models/text-embedding-004"
QDRANT_API_URL=qdrant_cloud_url
QDRANT_API_KEY=qdrant_cloud_api
TOKEN=your-openai-api-key
```

### Chạy ứng dụng
```plaintext
streamlit run main.py
```


## 🔍 Cơ chế hoạt động (RAG Pipeline)

Hệ thống thực hiện quy trình hỏi đáp dựa trên tài liệu pháp luật qua các bước sau:

1. **Document Loading**
   - Tài liệu pháp luật được tải lên và phân tích cú pháp.
   - Nội dung được chia thành các **Chương**, **Điều**, và các đoạn nhỏ (chunk) có ý nghĩa.

2. **Metadata Extraction**
   - Trích xuất thông tin quan trọng từ tài liệu, bao gồm:
     - Tên văn bản luật
     - Số hiệu
     - Ngày ban hành
     - Chương, Điều, Tiêu đề
   - Metadata giúp cải thiện khả năng tìm kiếm và truy xuất.

3. **Keyword Extraction & Embedding**
   - Sử dụng **Gemini API** để trích xuất từ khóa quan trọng từ mỗi đoạn.
   - Áp dụng kỹ thuật embedding để chuyển văn bản thành vector không gian ngữ nghĩa.

4. **Hybrid Retrieval**
   - **Bước 1: Keyword Search**  
     Lọc sơ bộ các đoạn văn bản chứa từ khóa liên quan đến câu hỏi.
   - **Bước 2: Semantic Search (Qdrant)**  
     Dùng embedding để tìm những đoạn văn bản có ngữ nghĩa gần với truy vấn nhất.

5. **LLM Generation**
   - Gửi các đoạn văn bản phù hợp vào mô hình ngôn ngữ **Gemini**.
   - Mô hình sinh ra câu trả lời tự nhiên và chính xác dựa trên ngữ cảnh cung cấp.

## 📚 Tài liệu tham khảo

- [LangChain Documentation](https://docs.langchain.com/)
- [Qdrant Documentation](https://qdrant.tech/documentation/)
- [PhoBERT (HuggingFace)](https://huggingface.co/VinAI/phobert-base)
- [VnCoreNLP (GitHub)](https://github.com/vncorenlp/VnCoreNLP)

---

## 📌 Ghi chú

- Dự án được thiết kế dành riêng cho các ứng dụng hỏi đáp dựa trên **văn bản pháp luật tiếng Việt**.
- Có thể mở rộng để áp dụng cho bất kỳ hệ thống RAG nào xử lý tiếng Việt, đặc biệt trong các lĩnh vực cần độ chính xác cao như: luật, giáo dục, y tế, v.v.

