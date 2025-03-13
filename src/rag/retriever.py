import os
from langchain_core.documents import Document
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, models
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

        # Khởi tạo embedding model (sử dụng Google Generative AI Embeddings)
        self.embedding_model = GoogleGenerativeAIEmbeddings(
            model=os.getenv('EMBEDDING_MODEL_NAME'),
            google_api_key=os.getenv('GEMINI_API_KEY')
        )
        
        # Khởi tạo Qdrant Client cho keyword searchu thuần túy
        self.qdrant_client = QdrantClient(
            url=os.getenv('QDRANT_API_URL'),
            api_key=os.getenv('QDRANT_API_KEY')
        )
        
        # Khởi tạo vector store với Qdrant cho semantic search 
        self.vector_store = QdrantVectorStore.from_existing_collection(
            embedding=self.embedding_model,
            url=os.getenv('QDRANT_API_URL'),
            api_key=os.getenv('QDRANT_API_KEY'),
            collection_name=self.collection_name,
            metadata_payload_key='metadata'
        )

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
        
        print(metadata_conditions)
        print(page_content_conditions)
        
        keyword_filter = models.Filter(
            must=metadata_conditions if metadata_conditions else None,
            min_should=models.MinShould(conditions=page_content_conditions, min_count=1) if page_content_conditions else None
        )
        
        # Tìm kiếm bằng Qdrant Client (không dùng vector)
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
        
        # # Thực hiện tìm kiếm theo keyword
        # keyword_results = self.vector_store.similarity_search_with_score(
        #     query=query,
        #     k=top_k,
        #     filter=keyword_filter if (metadata_conditions or page_content_conditions) else None
        # )
        
        # return keyword_results        

    def semantic_search(self, query: str, top_k: int = 20) -> List[Document]:
        """Chạy semantic search thuần dựa trên vector similarity (không có keyword filtering)"""
        semantic_results = self.vector_store.similarity_search_with_score(query=query, k=top_k)
        print(f"Semantic search results: {len(semantic_results)} docs")
        
        return [(doc, score) for doc, score in semantic_results]

    def hybrid_search(self, query: str, keywords: Dict, top_k: int = 10, use_ranker: bool = False) -> List[Document]:
        """Kết hợp keyword search và semantic search"""
        start_time = time.time()
        
        # Bước 1: Tìm kiếm ban đầu
        keyword_docs = self.keyword_search(keywords=keywords, top_k=top_k*2) # Lấy nhiều để rerank
        semantic_results = self.semantic_search(query=query, top_k=top_k*2)
        semantic_docs = [doc for doc, _ in semantic_results]
        semantic_scores = {doc.page_content: score for doc, score in semantic_results}
        
        # Bước 2: gộp kết quả, loại bỏ trùng lặp
        all_docs_dict = {doc.page_content: doc for doc in keyword_docs + semantic_docs}
        all_docs = list(all_docs_dict.values())
        print(f"Tổng chunk trước khi rerank: {len(all_docs)}")
        
        # Bước 3: Reranking
        if use_ranker: 
            # Dùng PhoRanker để rerank
            pass
        else: 
            # Dùng điểm tổng hợp đơn giản (keyword presence + semantic score)
            final_docs_with_scores = []
            for doc in all_docs:
                keyword_score = 1.0 if doc.page_content in [d.page_content for d in keyword_docs] else 0.0
                semantic_score = semantic_scores.get(doc.page_content, 0.0)
                total_score = 0.4*keyword_score + 0.6*semantic_score # Trọng số tùy chỉnh
                final_docs_with_scores.append((doc, total_score))
            
            # Sắp xếp theo điểm tổng hợp
            final_docs_with_scores.sort(key=lambda x: x[1], reverse=True)
            final_docs = [doc for doc, _ in final_docs_with_scores[:top_k]]
        
        
        # # Tìm kiếm bằng keyword
        # keyword_results = self.keyword_search(query=query, keywords=keywords, top_k=top_k)
        
        # # Tìm kiếm bằng semantic search
        # semantic_results = self.semantic_search(query=query, top_k=top_k)
        
        # # Tạo tập hợp nội dung từ keyword search và semantic search để kiểm tra trùng lặp
        # keywords_contents = {doc.page_content for doc, _ in keyword_results}
        # semantic_contents = {doc.page_content for doc, _ in semantic_results}
        
        # # Danh sách kết quả cuối cùng
        # final_results = []
        # seen_contents = set() # Sử dụng set để kiểm tra trùng lặp nhanh hơn
        
        # # 1.Ưu tiên các kết quả xuất hiện ở cả keyword_search và semantic_search
        # for keyword_doc, keyword_score in keyword_results:
        #     if keyword_doc.page_content in semantic_contents:
        #         final_results.append(keyword_doc)
        #         seen_contents.add(keyword_doc.page_content)
                
        # # 2. Thêm các kết quả chỉ có trong keyword_search (không trùng với semantic)
        # for keyword_docs, keyword_score in keyword_results:
        #     if keyword_doc.page_content not in seen_contents:
        #         final_results.append(keyword_doc)
        #         seen_contents.add(keyword_doc.page_content)

        # # 3. Nếu vẫn chưa đủ top_k, có thể bổ sung thêm từ semantich_search
        # for semantic_doc, semantic_score in semantic_results:
        #     if semantic_doc.page_content not in seen_contents and len(final_results) < top_k:
        #         final_results.append(semantic_doc)
        #         seen_contents.add(semantic_doc.page_content)
                
        end_time = time.time()
        excution_time = end_time - start_time
        print(f"Thời gian truy xuất: {excution_time:.4f} seconds")
        
        print(final_docs)
                
        return final_docs

# Test code
if __name__ == "__main__":
    retriever = Retriever()

    query = "phạm vi điều chỉnh của luật an toàn thông tin mạng số: 86/2015/QH13"
    extractor = KeywordsExtractor()
    keywords = extractor.extract_entities(query)
    
    results = retriever.keyword_search(keywords=keywords)
    # results = retriever.semantic_search(query=query)
    print(results)
    # # Test hybrid search
    print("Hybrid Search (simple scoring):")
    results_simple = retriever.hybrid_search(query=query, keywords=keywords, use_ranker=False)
    
    # print("\nHybrid Search (Cross-Encoder):")
    # results_cross = retriever.hybrid_search(query=query, keywords=keywords, use_cross_encoder=True)