import os 
from langchain_core.documents import Document
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableSequence, RunnableParallel
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from rich import print
from typing import List, Dict
import time

load_dotenv()

class Generator:
    def __init__(self, model: str = 'gemini-2.0-flash-exp', temperature: float = 0.0):
        """Khởi tạo Generator với model và temperature"""
        self.llm = ChatGoogleGenerativeAI(
            model=model,
            api_key=os.getenv('GEMINI_API_KEY'),
            temperature=temperature,
            max_tokens= 1024
        )
        self.prompt_template = ChatPromptTemplate.from_messages([
            (
              "system", 
              """Bạn là một trợ lý pháp lý thông minh.
              Dựa trên thông tin từ các tài liệu pháp lý dưới đây, hãy trả lời câu hỏi của người dùng một cách chính xác, ngắn gọn và dễ hiểu, chỉ tập trung vào phần liên quan trực tiếp đến câu hỏi.
              Không cung cấp thông tin thừa ngoài câu hỏi.
              Nếu không đủ thông tin để trả lời, hãy nói rằng không đủ thông tin và gợi ý tìm kiếm thêm."""  
            ),
            (
                "human",
                "Câu hỏi {question}\nContext: {context}\nTrả lời:"
            )
        ])
        
        # Pipeline RAG
        self.rag_chain = RunnableParallel(
            {
                "context": RunnablePassthrough(), # context sẽ được truyền từ retrieved_docs
                "question": RunnablePassthrough(), # question sẽ được truyền từ input
            }
        ) | RunnableSequence(
            self._format_context, # tổng hợp context từ các tài liệu được truy xuất
            self.prompt_template, # tạo prompt từ context và question
            self.llm, # tạo trả lời từ prompt
            StrOutputParser() # parse output từ trả lời
        ) 
    
    def _format_context(self, inputs: Dict) -> Dict:
        """Định dạng context từ danh sách tài liệu và câu hỏi"""
        docs = inputs['context']
        if not docs:
            formatted_context = "Không có thông tin nào được cung cấp."
        else:
            formatted_context = "\n\n".join(f"[{i+1}] {doc.page_content}" for i, doc in enumerate(docs) if isinstance(doc, Document))
            return {
                "context": formatted_context,
                "question": inputs['question']
            }
            
    def generate_answer(self, question: str, retrieved_docs: List[Document]) -> str:
        """Tạo trả lời từ câu hỏi và danh sách tài liệu truy xuất"""
        try: 
            answer = self.rag_chain.invoke({
                "question": question,
                "context": retrieved_docs
            })
        except Exception as e:
            answer = f"Lỗi: {str(e)}"
        
        return answer
    
