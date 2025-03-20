import os
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from google import genai
from google.genai.types import EmbedContentConfig 
from typing import List, Dict
from dotenv import load_dotenv
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, '..')
sys.path.insert(0, parent_dir)
import time 
from rich import print
from rich import traceback
from sentence_transformers import CrossEncoder

from src.processor.keyword_extractor import KeywordsExtractor

traceback.install()
load_dotenv()

class Retriever:
    def __init__(self, collection_name: str = "hpt_rag_pipeline"):
        """Khởi tạo kết nối với Qdrant và chuẩn bị vectorstore"""
        self.collection_name = collection_name
        
        self.embedding_model = genai.Client(api_key=os.getenv('GEMINI_API_KEY'))

        # # Khởi tạo embedding model (sử dụng Google Generative AI Embeddings)
        # self.embedding_model = GoogleGenerativeAIEmbeddings(
        #     model=os.getenv('EMBEDDING_MODEL_NAME'),
        #     google_api_key=os.getenv('GEMINI_API_KEY'),
        #     task_type="retrieval_query"
        # )
        
        # Khởi tạo Qdrant Client cho keyword search thuần túy
        self.qdrant_client = QdrantClient(
            url=os.getenv('QDRANT_API_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        # # Khởi tạo vector store với Qdrant cho semantic search 
        # self.vector_store = QdrantVectorStore.from_existing_collection(
        #     embedding=self.embedding_model,
        #     url=os.getenv('QDRANT_API_URL'),
        #     api_key=os.getenv('QDRANT_API_KEY'),
        #     collection_name=self.collection_name,
        #     metadata_payload_key='metadata'
        # )
        
    def embed_text(self, text:str) -> List[float]:
        response = self.embedding_model.models.embed_content(
            model=os.getenv('EMBEDDING_MODEL_NAME'),
            contents=[text],
            config=EmbedContentConfig(
                task_type='SEMANTIC_SIMILARITY',
                output_dimensionality=768
            )
        )
        
        return response.embeddings[0].values

    def keyword_search(self, keywords: Dict, top_k: int = 20) -> List[Document]:
        """
        Tìm kiếm chỉ với keyword filtering trên page_content và metadata 
        (Không dùng vector similarity)
        """
        print(keywords)

        # Tạo danh sách điều kiện
        metadata_conditions = []
        page_content_conditions = []
        
        # Kiểm tra và thêm điều kiện cho các trường metadata
        for field in ['type', 'title', 'number', 'issued_date', 'chapter', 'section', 'article']:
            value = keywords.get(field, "")
            if value and value != "":
                if field == "article":
                    # Chuẩn hóa giá trị tìm kiếm cho article
                    article_value = f"điều {value}" if value.isdigit() else value.lower()
                    metadata_conditions.append(
                        models.FieldCondition(
                            key=f'metadata.{field}',
                            match=models.MatchValue(value=article_value)
                        )
                    )
                else:
                    metadata_conditions.append(
                        models.FieldCondition(
                            key=f'metadata.{field}',
                            match=models.MatchText(text=value.lower())
                        )
                    )
        
        # Thêm điều kiện cho keywords trong page_content
        for keyword in keywords.get('keywords', []):
            if keyword and keyword.strip():
                page_content_conditions.append(
                    models.FieldCondition(
                        key='page_content',
                        match=models.MatchText(text=keyword.strip())
                    )
                )
        
        # print(metadata_conditions)
        # print(page_content_conditions)
        
        keyword_filter = models.Filter(
            must=metadata_conditions if metadata_conditions else None,
            min_should=models.MinShould(conditions=page_content_conditions, min_count=1) if page_content_conditions else None
        )
        
        # Tìm kiếm đồng bộ bằng Qdrant Client (không dùng vector)
        keyword_results = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            scroll_filter=keyword_filter,
            limit=top_k,
            with_payload=True,
            with_vectors=False
        )
        
        # Chuyển đổi kết quả thành Document
        keyword_docs = []
        for point in keyword_results[0]:
            metadata = point.payload.get("metadata", {})
            page_content = point.payload.get("page_content", '')
            keyword_docs.append(
                Document(
                    page_content=page_content,
                    metadata=metadata
                )
            )
            
        print(f"Keyword search results: {len(keyword_docs)} docs")
        return keyword_docs    

    def semantic_search(self, query: str, top_k: int = 20) -> List[Document]:
        """Chạy semantic search thuần dựa trên vector similarity (không có keyword filtering)"""
        # semantic_results = self.vector_store.similarity_search_with_score(query=query, k=top_k)
        
        # print(f"Semantic search results: {len(semantic_results)} docs")
        
        # return [(doc, score) for doc, score in semantic_results]
        
        query_embedding = self.embed_text(query)
        search_result = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k,
            with_payload=True
        )
        
        print(search_result)
        
        return [
            Document(
                page_content=hit.payload.get('page_content', ''),
                metadata = hit.payload.get('metadata', {}),
            )
            for hit in search_result
        ]

    def hybrid_search(self, query: str, keywords: Dict, top_k: int = 5, use_ranker: bool = False) -> List[Document]:
        """Kết hợp keyword search và semantic search"""
        start_time = time.time()
        
        # Bước 1: Tìm kiếm ban đầu
        keyword_docs = self.keyword_search(keywords=keywords, top_k=top_k)  # Lấy nhiều để rerank
        
        semantic_results = self.semantic_search(query=query, top_k=top_k)
        
        semantic_docs = [doc for doc, _ in semantic_results]
        
        # Bước 2: Gộp kết quả, loại bỏ trùng lặp
        all_docs_dict = {doc.page_content: doc for doc in keyword_docs + semantic_docs}
        all_docs = list(all_docs_dict.values())
        print(f"Tổng chunk trước khi rerank: {len(all_docs)}")
        
        # Bước 3: Reranking với Weighted RRF
        if use_ranker: 
            # Dùng PhoRanker để rerank (chưa triển khai)
            pass
        else: 
            # Dùng điểm tổng hợp đơn giản (keyword presence + semantic score) dựa trên công thức RRF
            rrf_scores = {}
            k = 60
            
            # Tính rank cho keyword search
            for rank, doc in enumerate(keyword_docs, 1):
                content = doc.page_content
                if content not in rrf_scores:
                    rrf_scores[content] = 0
                rrf_scores[content] += 0.4 * (1 / (k + rank))
                    
            # Tính rank cho semantic search
            for rank, (doc, _) in enumerate(semantic_results, 1):
                content = doc.page_content
                if content not in rrf_scores:
                    rrf_scores[content] = 0
                rrf_scores[content] += 0.6 * (1 / (k + rank))
                
            # Sắp xếp tài liệu theo RRF
            final_docs_with_scores = [(doc, rrf_scores.get(doc.page_content, 0)) for doc in all_docs]
            final_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in final_docs_with_scores[:top_k]]
            
        end_time = time.time()
        excution_time = end_time - start_time
        print(f"Thời gian truy xuất: {excution_time:.4f} seconds")
        
        print(final_docs)
                
        return final_docs

def main():
    retriever = Retriever()

    query = "hồ sơ phê duyệt đề xuất cấp độ bao gồm những gì?"

    # extractor = KeywordsExtractor()
    # keywords = extractor.extract_entities(query)
    
    # print("Keyword Search:")
    # results = retriever.keyword_search(keywords=keywords)
    # print(results)

    print("Semantic Search:")
    results = retriever.semantic_search(query=query)
    print(results)
    
    # # Test hybrid search
    # print("\nHybrid Search (Weighted RRF):")
    # results_hybrid = retriever.hybrid_search(query=query, keywords=keywords, use_ranker=False)

# Test code
if __name__ == "__main__":
    main()