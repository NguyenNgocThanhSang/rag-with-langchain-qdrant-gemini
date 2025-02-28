import streamlit as st
from src.rag.retriever import Retriever
from src.rag.generator import Generator
from src.processor.keyword_extractor import KeywordsExtractor
import os
from dotenv import load_dotenv

# Tải biến môi trường từ file .env
load_dotenv()

# Khởi tạo các thành phần RAG
retriever = Retriever()
generator = Generator(temperature=0.9)  # Sử dụng temperature như trong code của bạn
extractor = KeywordsExtractor()

# Cấu hình trang Streamlit với độ rộng tối đa
st.set_page_config(
    page_title="RAG Demo: Hỏi đáp với tài liệu",
    page_icon="🦜",
    layout="wide"  # Sử dụng layout "wide" để mở rộng theo chiều ngang
)

st.title("🦜 RAG Demo: Hỏi đáp với tài liệu")

# Ô nhập truy vấn
query = st.text_input("Nhập câu hỏi của bạn:", placeholder="Ví dụ: Hành vi sử dụng thông tin, dữ liệu khí tượng thủy văn không đúng mục đích bị phạt bao nhiêu tiền?")

if st.button("Gửi"):
    if query:
        st.info("Đang xử lý câu hỏi...")
        
        # Trích xuất từ khóa
        keywords = extractor.extract_entities(query)
        st.write("Từ khóa trích xuất:")
        st.json(keywords)  # Hiển thị từ khóa dưới dạng JSON để dễ đọc

        # Tìm kiếm tài liệu bằng retriever (hybrid search)
        retrieved_docs = retriever.hybrid_search(query=query, keywords=keywords)
        
        if not retrieved_docs:
            st.warning("Không tìm thấy tài liệu liên quan.")
        else:
            # Tạo câu trả lời bằng generator
            answer = generator.generate_answer(question=query, retrieved_docs=retrieved_docs)
            st.success("Câu trả lời:")
            st.write(answer)

# Hiển thị thông tin bổ sung
st.markdown("---")
st.write("Ứng dụng RAG sử dụng Retriever, Generator và KeywordsExtractor để trả lời dựa trên tài liệu trong Qdrant.")