# main.py
import streamlit as st
from src.rag.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os

# Tải biến môi trường từ file .env
load_dotenv()

# Khởi tạo RAG Pipeline
@st.cache_resource
def initialize_rag():
    rag = RAGPipeline(
        collection_name="legal_docs",
        model=os.getenv("MODEL_NAME", "gemini-2.0-flash-exp")
    )
    return rag

def main():
    # Thiết lập cấu hình trang
    st.set_page_config(
        page_title="RAG Demo",
        page_icon="🦜",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Khởi tạo RAG nếu chưa có
    if "rag" not in st.session_state:
        with st.spinner("Đang khởi tạo hệ thống..."):
            st.session_state.rag = initialize_rag()
    rag = st.session_state.rag

    # Tạo sidebar
    with st.sidebar:
        st.header("Chatbot")
        "[View the source code](https://github.com/streamlit/llm-examples/blob/main/Chatbot.py)"
        "[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/streamlit/llm-examples?quickstart=1)"

    # Tiêu đề và mô tả
    st.title("💬 Chatbot")
    st.caption("🚀 A Streamlit chatbot powered by Gemini")

    # Khởi tạo lịch sử chat
    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "How can I help you?"}]

    # Hiển thị lịch sử chat
    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # Nhập câu hỏi từ người dùng
    if prompt := st.chat_input("Your message"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").write(prompt)
        
        # Sử dụng RAGPipeline để tạo câu trả lời
        with st.spinner("Đang xử lý..."):
            response = rag.run(query=prompt, top_k=5)
            st.session_state.messages.append({"role": "assistant", "content": response})
            st.chat_message("assistant").write(response)

if __name__ == "__main__":
    main()