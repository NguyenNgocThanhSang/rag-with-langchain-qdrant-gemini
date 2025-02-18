import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_google_genai import GoogleGenerativeAIEmbeddings, GoogleGenerativeAI
from qdrant_client.models import VectorParams, Distance
from typing import List
from dotenv import load_dotenv()

load_dotenv()

class Retriever:
    def __init__(self, collection_name: str="legal_docs"):
        self.collection_name = collection_name
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedding_model,
            url=os.getenv('QDRANT_API_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=self.collection_name
        )
        
    # def keyword_search(self, keywords: List[str], )
