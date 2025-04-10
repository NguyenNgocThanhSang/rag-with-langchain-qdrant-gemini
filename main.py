# main.py
import streamlit as st
from src.rag.rag_pipeline import RAGPipeline
from dotenv import load_dotenv
import os
import time
import torch
from rich import traceback
traceback.install()

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]


# Tải biến môi trường từ file .env
load_dotenv()

# Khởi tạo RAG Pipeline
@st.cache_resource
def initialize_rag():
    rag = RAGPipeline(
        collection_name="hpt_rag_pipeline",
        gemini_model=os.getenv("MODEL_NAME", "gemini-2.0-flash-exp"),
        openai_model="llm-large-v4"
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
    st.caption("🚀 A Streamlit chatbot comparing Gemini and OpenAI")

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
        
        # Sử dụng RAGPipeline để tạo câu trả lời từ cả hai mô hình
        with st.spinner("Đang xử lý..."):
            start_time = time.time()
            response = rag.run(query=prompt, top_k=5)
            end_time = time.time()
            elapsed_time = end_time - start_time
            
            # Lưu và hiển thị câu trả lời từ Gemini
            gemini_response = response["Gemini"]
            st.session_state.messages.append({"role": "assistant", "content": f"Gemini: {gemini_response}"})
            with st.chat_message("assistant"):
                st.markdown("**Gemini:**")
                st.write(gemini_response)
            
            # Lưu và hiển thị câu trả lời từ OpenAI
            openai_response = response["OpenAI"]
            st.session_state.messages.append({"role": "assistant", "content": f"OpenAI: {openai_response}"})
            with st.chat_message("assistant"):
                st.markdown("**OpenAI:**")
                st.write(openai_response)
            
            # Hiển thị thời gian xử lý
            st.caption(f"Thời gian xử lý: {elapsed_time:.2f} giây")

if __name__ == "__main__":
    main()