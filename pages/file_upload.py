import streamlit as st
from src.processor.document_loader import DocumentLoader
from src.processor.vector_store import QdrantDatabase
from uuid import uuid4
from rich import print
from rich import traceback
import torch
import os
import time

torch.classes.__path__ = [os.path.join(torch.__path__[0], torch.classes.__file__)]

traceback.install()

# Khởi tạo các đối tượng global
@st.cache_resource
def init_qdrant():
    return QdrantDatabase()

vector_store = init_qdrant()

# Giao diện chính
st.title("📝 Upload file to Qdrant")
uploaded_file = st.file_uploader("Upload an document", type=("docx"))

# Xử lý upload file
if uploaded_file:
    try:
        start_time = time.time()
        # Tạo filepath tạm thời
        temp_file_path = uploaded_file.name
        with open(temp_file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        # Load và split document 
        loader = DocumentLoader(file_path=temp_file_path)
        documents = loader.load_and_split()
        
        # Upload lên Qdrant
        with st.spinner("Đang tải lên Qdrant..."):
            vector_store.upload(documents=documents)
        end_time = time.time()
        elapsed_time = end_time - start_time
        st.success(f"Tải tài liệu lên thành công! Thời gian xử lý: {elapsed_time:.2f} giây")
    
    except Exception as e:
        st.error(f"Lỗi xử lý file: {str(e)}")
        
    finally:
        # Xóa file tạm
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            
