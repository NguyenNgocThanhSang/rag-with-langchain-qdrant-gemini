import os
from langchain_qdrant import QdrantVectorStore
from qdrant_client import models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

class Retriever:
    def __init__(self, collection_name: str = "legal_docs"):
        """Khởi tạo kết nối với Qdrant và chuẩn bị vectorstore"""
        self.collection_name = collection_name

        # Khởi tạo embedding model (sử dụng Google Generative AI Embeddings)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Khởi tạo vector store với Qdrant
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedding_model,
            url=os.getenv('QDRANT_API_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=self.collection_name,
            metadata_payload_key='metadata'
        )

    def keyword_search(self, query: str, keywords: List[str], top_k: int = 20) -> List[Dict]:
        """Tìm kiếm chỉ với keyword filtering trên page_content và metadata"""
        keyword_filter = models.Filter(
            should=[
                models.FieldCondition(
                    key='metadata.article',
                    match=models.MatchValue(value="Điều 79"))
                # ) for keyword in keywords
            ] 
            # [
            #     models.FieldCondition(
            #         key='metadata',
            #         match=models.MatchValue(value=keyword)
            #     ) for keyword in keywords
            # ],
        )
        
        # print(keyword_filter)

        # Thực hiện tìm kiếm theo keyword
        keyword_results = self.vector_store.similarity_search_with_score(
            query="",  # Không cần query, chỉ sử dụng filter
            k=top_k,
            filter=
        )

        # In kết quả debug nếu cần
        return [
            {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            }
            for doc, score in keyword_results
        ]

    def semantic_search(self, query: str, top_k: int = 4) -> List[Dict]:
        """Chạy semantic search thuần (không có keyword filtering)"""
        semantic_results = self.vector_store.similarity_search_with_score(query=query, k=top_k)

        return [
            {
                'text': doc.page_content,
                'metadata': doc.metadata,
                'score': score
            }
            for doc, score in semantic_results
        ]

    def hybrid_search(self, query: str, keywords: List[str], top_k: int = 20) -> List[Dict]:
        """Kết hợp keyword search và semantic search"""
        # Tìm kiếm bằng keyword
        keyword_results = self.keyword_search(query, keywords, top_k)

        # Tìm kiếm bằng semantic search
        semantic_results = self.semantic_search(query, top_k)

        # Kết hợp kết quả của cả 2 phương pháp
        return keyword_results + semantic_results


# Test code
if __name__ == "__main__":
    retriever = Retriever()

    query = "Điều 79. Hoạt động vận tải đường bộ trong đô thị"    
    keywords = ['Điều 79', 'Hoạt động vận tải đường bộ trong đô thị', 'Xe buýt phải chạy đúng tuyến', 'Xe vệ sinh môi trường']

    print("Keyword Search:", retriever.keyword_search(query, keywords), '\n')
    # print("Semantic Search:", retriever.semantic_search(query), '\n')
    # print("Hybrid Search:", retriever.hybrid_search(query, keywords))
