import os
from langchain_qdrant import QdrantVectorStore, RetrievalMode
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from google.genai.types import EmbedContentConfig 
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
from typing import List
from langchain_core.documents import Document
from uuid import uuid4
from dotenv import load_dotenv
import time

load_dotenv()

class QdrantDatabase:
    def __init__(self, collection_name:str = 'hpt_rag_pipeline', vector_size: int = 768):
        """Khởi tạo kết nối với Qdrant và xác định collection để lưu dữ liệu"""
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.client = QdrantClient(
            url=os.getenv('QDRANT_API_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        self.embedding_model = genai.Client(
            api_key=os.getenv('GEMINI_API_KEY'),
        )
        
        # self.embedding_model=GoogleGenerativeAIEmbeddings(
        #     model=os.getenv('EMBEDDING_MODEL_NAME'),
        #     google_api_key=os.getenv('GEMINI_API_KEY'),
        #     task_type="retrieval_document",
        #     output_dimensionality = self.vector_size
        # )
        
    def _ensure_collection_exists(self):
        """Kiểm tra xem collection có chưa, chưa có thì tạo mới"""
        collections = self.client.get_collections()
        
        if self.collection_name not in [collection.name for collection in collections.collections]:
            print(f"collection {self.collection_name} chưa tồn tại, đang tạo mới")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )
        else:
            print(f"Collection {self.collection_name} đã tồn tại.")
            
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Nhúng tài liệu thành embeddings"""
        response = self.embedding_model.models.embed_content(
            model=os.getenv('EMBEDDING_MODEL_NAME'),
            contents=texts,
            config=EmbedContentConfig(
                task_type='RETRIEVAL_DOCUMENT',
                output_dimensionality=self.vector_size
            )
        )
        
        return [embedding.values for embedding in response.embeddings]
    
    def upload(self, documents: List[Document]):
        """Nhúng và lưa các đoạn chunk vào Qdrant"""
        # embeded_texts = self.embed_documents(texts=texts)
        # Kiểm tra và tạo collection nếu chưa tồn tại
        self._ensure_collection_exists()
        
        # # Tạo vector_store và upload Documents 
        # vector_store = QdrantVectorStore(
        #     client=self.client,
        #     collection_name=self.collection_name,
        #     embedding=self.embedding_model,
        #     # retrieval_mode=RetrievalMode.HYBRID,
        # )
        
        # uuids = [str(uuid4()) for _ in range(len(documents))]
        # vector_store.add_documents(documents=documents, ids=uuids)
        # print("Upload tài liệu thành công")
        
        texts = [doc.page_content for doc in documents]
        embedded_texts = self.embed_documents(texts=texts)
        uuids = [str(uuid4()) for _ in range(len(documents))]
        payloads = [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata
            }
            for doc in documents
        ]
        
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=uuid,
                    vector=vector,
                    payload=payload
                ) for uuid, vector, payload in zip(uuids, embedded_texts, payloads)
            ]
        )
        
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
