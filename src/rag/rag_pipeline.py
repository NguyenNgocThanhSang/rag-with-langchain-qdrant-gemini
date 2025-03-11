import os
from typing import List, Dict
from langchain_core.documents import Document
from dotenv import load_dotenv

# Import modules Retriever and Generator
from .generator import Generator
from .retriever import Retriever
from src.processor.keyword_extractor import KeywordsExtractor

load_dotenv()

class RAGPipeline:
    """Pipeline RAG cho hệ thống hỏi đáp với tài liệu pháp lý"""
    def __init__(self, collection_name: str = 'hpt_rag_pipeline', model: str = os.getenv("MODEL_NAME")):
        """Khởi tạo RAG Pipeline với Retriever và Generator"""

        # Khởi tạo Retriever
        self.retriever = Retriever(collection_name=collection_name)
        
        # Khởi tạo Generator với mô hình Gemini
        self.generator = Generator(model=model)
        
        # Khởi tạo KeywordsExtractor
        self.keywords_extractor = KeywordsExtractor()
        
    def run(self, query: str, keywords: Dict = None, top_k: int = 10) -> str:
        """
        Chạy pipeline RAG: truy xuất tài liệu và tạo câu trả lời
        
        Args: 
            query(str): Câu hỏi từ người dùng
            
        Returns:
            str: Câu trả lời từ Generator
        """
        try:
            # Trích xuất từ khóa từ câu hỏi
            keywords = self.keywords_extractor.extract_entities(query)
            
            # Truy xuất tài liệu từ Qdrant sử dụng hybrid search
            retrieved_docs = self.retriever.hybrid_search(query=query, keywords=keywords, top_k=top_k)

            # Tạo câu trả lời từ Generator
            answer = self.generator.generate_answer(question=query, retrieved_docs=retrieved_docs)
            
            return answer
        
        except Exception as e:
            return f"Lỗi trong Pipeline RAG: {str(e)}"
        
# if __name__ == "__main__":
#     pipeline = RAGPipeline()
    
#     query = "Hành vi sử dụng thông tin, dữ liệu khí tượng thủy văn không đúng mục đích bị phạt bao nhiêu tiền?"
    
#     response = pipeline.run(query=query)
    
#     print(response)