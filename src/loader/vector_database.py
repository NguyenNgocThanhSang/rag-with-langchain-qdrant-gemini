import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams
from typing import List, Dict
from langchain_core.documents import Document
from uuid import uuid4
from dotenv import load_dotenv
import time

load_dotenv()

class QdrantDatabase:
    def __init__(self, collection_name:str = 'legal_docs'):
        """Khởi tạo kết nối với Qdrant và xác định collection để lưu dữ liệu"""
        self.collection_name = collection_name
        self.client = QdrantClient(
            url=os.getenv('QDRANT_API_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        self.embedding_model=GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Kiểm tra collection xem có chưa, nếu chưa có thì tạo mới
        # self._ensure_collection_exists()
        
    def _ensure_collection_exists(self):
        """Kiểm tra xem collection có chưa, chưa có thì tạo mới"""
        collections = self.client.get_collections()
        
        if self.collection_name not in [collection.name for collection in collections.collections]:
            print(f"collection {self.collection_name} chưa tồn tại, đang tạo mới")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=768, distance=Distance.COSINE)
            )
        else:
            print(f"Collection {self.collection_name} đã tồn tại.")
            
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Nhúng tài liệu thành embeddings"""
        return self.embedding_model.embed_documents(texts)
    
    def upload(self, documents: List[Document]):
        """Nhúng và lưa các đoạn chunk vào Qdrant"""
        # embeded_texts = self.embed_documents(texts=texts)
        # Kiểm tra và tạo collection nếu chưa tồn tại
        self._ensure_collection_exists()
        
        # Tạo vector_store và upload Documents 
        vector_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.collection_name,
            embedding=self.embedding_model,
            # retrieval_mode=RetrievalMode.HYBRID,
        )
        
        uuids = [str(uuid4()) for _ in range(len(documents))]
        vector_store.add_documents(documents=documents, ids=uuids)
        print("Upload tài liệu thành công")
    
    def delete_collection(self):
        '''Xóa toàn bộ collection'''
        collections = self.client.get_collections()
        
        if self.collection_name in [collection.name for collection in collections.collections]:
            self.client.delete_collection(collection_name=self.collection_name)
            print(f'{self.collection_name} đã bị xóa')
            # Đảm bảo collection đã bị xóa hoàn toàn
            time.sleep(1)
            collections = self.client.get_collections()
            if self.collection_name not in [collection.name for collection in collections.collections]:
                print(f"{self.collection_name} đã bị xóa hoàn toàn.")
            else:
                print(f"Lỗi: {self.collection_name} vẫn tồn tại sau khi xóa.")
        else:
            print(f'{self.collection_name} không tồn tại, không cần xóa')
