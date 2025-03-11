from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from typing import List, Dict, Optional
from pydantic import BaseModel, Field
import os
import json
from rich import print
from rich import traceback
from dotenv import load_dotenv

traceback.install()
load_dotenv()

# Định nghĩa Pydantic schema
class Entity(BaseModel):
    """Mô tả các keywords được trích xuất từ truy vấn"""
    type: str = Field(
        default="",
        description="Loại văn bản, ví dụ: 'luật', 'quyết định', 'thông tư'"
    )
    title: str = Field(
        default="",
        description="Tiêu đề văn bản, ví dụ 'luật giao thông đường bộ'"
    )
    number: str = Field(
        default="",
        description="Số hiệu văn bản, ví dụ: '01/2021/qđ-ubnd'"
    )
    issued_date: str = Field(
        default="",
        description="Thời gian ban hành, ví dụ: '01/01/2025', '2025', '01/2025'"
    )
    chapter: str = Field(
        default="",
        description="Chương, ví dụ: 'chương i', 'chương ii'"
    )
    section: str = Field(
        default="",
        description="Mục, ví dụ: 'mục 1', 'mục 2'"
    )
    article:str = Field(
        default="",
        description="Điều luật, ví dụ: 'điều 1', 'điều 2'"
    )
    keywords: List[str] = Field(
        default_factory=list,
        description="Danh sách các từ khóa quan trọng, ví dụ: ['giao thông', 'đường bộ']"
    )

class KeywordsExtractor:
    '''Lớp trích xuất thực thể (keywords) từ truy vấn người dùng'''
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv('MODEL_NAME'),
            api_key=os.getenv('GEMINI_API_KEY'),
        )
        self.prompt_template = PromptTemplate(
            input_variables=['query'],
            template="""
            Bạn là một trợ lý AI chuyên xử lý văn bản pháp luật tiếng Việt. Dựa trên câu sau, trích xuất các thực thể quan trọng và trả về theo cấu trúc đã định nghĩa.

            Câu query: "{query}"

            Trả lời bằng cách cung cấp các thông tin sau (nếu không có, bỏ trống):
            - type: loại văn bản.
            - title: tiêu đề văn bản.
            - number: số hiệu văn bản.
            - issued_date: thời gian ban hành.
            - chapter: chương.
            - section: mục.
            - article: điều luật.
            - keywords: danh sách các từ khóa quan trọng.

            Lưu ý: Giữ nguyên kiểu chữ hoa/thường từ câu query đầu vào.
            """
        )
        
    def extract_entities(self, query: str) -> Dict:
        """
        Trích xuất thực thể (keywords) từ truy vấn người dùng 
        """
        if not query or not query.strip():
            raise ValueError("Truy vấn không được để trống")
        
        prompt_text = self.prompt_template.format(query=query)
        # print(prompt_text)
        response = self.llm.invoke(prompt_text)
        # print(response)
        
        # Trích xuất thông tin từ response
        try:
            content = response.content.strip()
            # Nếu có dấu code block, loại bỏ chúng:
            if content.startswith("```json"):
                content = content[len("```json"):].strip()
            if content.endswith("```"):
                content = content[:-3].strip()

            return json.loads(content)
            
        except Exception as e:
            raise Exception("Lỗi khi parse response: ", str(e))
        
    def dict_to_json(self, data: Dict) -> str:
        """
        Chuyển đổi một dictionary sang chuỗi JSON định dạng.
        
        Return:
            Chuỗi JSON đã được định dạng (với indent) và đảm bảo hiển thị tiếng Việt đúng (ensure_ascii=False).
        """
        try:
            return json.dumps(data, ensure_ascii=False, indent=4)
        except Exception as e:
            raise Exception("Lỗi khi chuyển đổi Dict sang JSON: " + str(e))

# # Test    
# if __name__ == "__main__":
#     extractor = KeywordsExtractor()
#     query = "điều 3 của luật giao thông đường bộ có những nội dung gì?"
#     entities = extractor.extract_entities(query.lower())
#     print("Parsed Entities:", entities)
    
#     json_string = extractor.dict_to_json(entities)
#     print("JSON String:\n", json_string)
    
    